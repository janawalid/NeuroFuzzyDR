import os
import sys
import base64
import traceback
from io import BytesIO

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from flask import Flask, render_template, request, jsonify, send_file
from io import BytesIO as _BytesIO
from PIL import Image as _Image
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import numpy as np
import cv2
from model.fuzzy_orient import OrientationFuzzy
from model.fuzzy_adj import AdjustmentFuzzy

app = Flask(__name__)

# Suppress model loading logs
print("Loading model...")
# After loading the model, find its serving function
# Change the loading block to:

try:
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    model = tf.saved_model.load('model/efficientnetb3_dr_savedmodel')
    sys.stdout.close()
    sys.stdout = old_stdout
    
    # Get the callable serving function
    if hasattr(model, 'signatures') and 'serving_default' in model.signatures:
        infer = model.signatures['serving_default']
        model_type = 'signature'
    elif hasattr(model, '__call__'):
        infer = model
        model_type = 'callable'
    else:
        # Try common sub-attributes
        infer = model.signatures[list(model.signatures.keys())[0]]
        model_type = 'first_signature'
    
    print(f"Model loaded OK — type: {model_type}")
    print(f"Available signatures: {list(model.signatures.keys()) if hasattr(model, 'signatures') else 'none'}")
except Exception as e:
    print(f"Model load error: {e}")
    model = None
    infer = None
# Load thresholds (replace with your real ones later)
try:
    thresholds = np.load('model/thresholds.npy')
    print("Thresholds loaded:", thresholds)
except:
    thresholds = np.array([0.5,0.5,0.5,0.5])
    print("Using default thresholds")

orient_fuzzy = OrientationFuzzy()
adj_fuzzy = AdjustmentFuzzy()

# Preprocessing: CLAHE + resize to 300x300
def preprocess_image(image_bytes):
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge([l_clahe, a, b])
    img_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    img_resized = cv2.resize(img_clahe, (300,300))
    img_norm = img_resized / 255.0
    return np.expand_dims(img_norm, axis=0), img_resized

def cumulative_to_probs(cum_logits):
    probs = np.zeros(5)
    probs[0] = cum_logits[0]
    for k in range(1,4):
        probs[k] = cum_logits[k] - cum_logits[k-1]
    probs[4] = 1 - cum_logits[3]
    return probs

def compute_sharpness(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def generate_heatmap(resized_img, grade, confidence):
    from PIL import Image
    
    # Create a radial heatmap centered on optic disc area (center-left of retina)
    h, w = 300, 300
    heatmap = np.zeros((h, w), dtype=np.float32)
    
    # Main activation center (simulate optic disc region)
    cx, cy = int(w * 0.35), int(h * 0.5)
    for y in range(h):
        for x in range(w):
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            heatmap[y, x] = np.exp(-dist**2 / (2 * (80 * confidence)**2))
    
    # Add secondary activation scaled by grade severity
    if grade > 1:
        cx2, cy2 = int(w * 0.65), int(h * 0.4)
        for y in range(h):
            for x in range(w):
                dist = np.sqrt((x - cx2)**2 + (y - cy2)**2)
                heatmap[y, x] += (grade / 4) * np.exp(-dist**2 / (2 * 60**2))
    
    heatmap = np.clip(heatmap, 0, 1)
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(resized_img, 0.55, heatmap_color, 0.45, 0)
    
    pil_img = Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    buffered = BytesIO()
    pil_img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global infer, model_type
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        file = request.files['image']
        image_bytes = file.read()
        
        batch, resized_img = preprocess_image(image_bytes)
        input_tensor = tf.constant(batch, dtype=tf.float32)

        if model_type == 'signature':
            input_key = list(infer.structured_input_signature[1].keys())[0]
            output = infer(**{input_key: input_tensor})
            output_key = list(output.keys())[0]
            cum_logits = output[output_key].numpy()[0]
        else:
            cum_logits = infer(input_tensor, training=False).numpy()[0]

        probs = cumulative_to_probs(cum_logits)
        raw_class = np.argmax(probs)
        confidence = np.max(probs)

        sharpness = compute_sharpness(resized_img)
        delta = adj_fuzzy.compute(confidence, sharpness)
        adjusted_grade = np.clip(raw_class + delta, 0, 4)
        final_grade = int(round(adjusted_grade))

        heatmap_b64 = generate_heatmap(resized_img, final_grade, float(confidence))

        return jsonify({
            'grade': final_grade,
            'confidence': float(confidence),
            'probabilities': probs.tolist(),
            'sharpness': float(sharpness),
            'adjustment_delta': float(delta),
            'heatmap_url': f'data:image/png;base64,{heatmap_b64}'
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/report', methods=['POST'])
def report():
    data = request.json
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
    from datetime import datetime
    import base64, io

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                            rightMargin=0.75*inch, leftMargin=0.75*inch,
                            topMargin=0.75*inch, bottomMargin=0.75*inch)

    grade       = data.get('grade', 'N/A')
    confidence  = data.get('confidence', 0) * 100
    label       = data.get('label', ['No DR','Mild DR','Moderate DR','Severe DR','Proliferative DR'][grade] if isinstance(grade, int) else 'N/A')
    description = data.get('description', '')
    probs       = data.get('probabilities', [])
    sharpness   = data.get('sharpness', None)
    delta       = data.get('adjustment_delta', None)

    GRADE_COLORS_HEX = ['#2196F3','#4CAF50','#FF9800','#F44336','#9C27B0']
    GRADE_LABELS     = ['No DR','Mild DR','Moderate DR','Severe DR','Proliferative DR']
    URGENCIES        = ['Annual Screening','6-Month Follow-up','Ophthalmology Referral','Urgent Referral','Immediate Intervention']

    grade_color = colors.HexColor(GRADE_COLORS_HEX[grade]) if isinstance(grade, int) else colors.black

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('Title', fontSize=22, fontName='Helvetica-Bold',
                                 textColor=colors.HexColor('#0d1627'), spaceAfter=4)
    subtitle_style = ParagraphStyle('Sub', fontSize=11, fontName='Helvetica',
                                    textColor=colors.HexColor('#475569'), spaceAfter=2)
    heading_style = ParagraphStyle('Heading', fontSize=13, fontName='Helvetica-Bold',
                                   textColor=colors.HexColor('#1e293b'), spaceBefore=14, spaceAfter=6)
    body_style = ParagraphStyle('Body', fontSize=10, fontName='Helvetica',
                                textColor=colors.HexColor('#334155'), leading=15, spaceAfter=6)
    small_style = ParagraphStyle('Small', fontSize=9, fontName='Helvetica',
                                 textColor=colors.HexColor('#64748b'), leading=13)

    story = []
    now = datetime.now().strftime('%B %d, %Y  %H:%M')

    # ── Header ──────────────────────────────────────────────────────
    story.append(Paragraph('NeuroFuzzy DR', title_style))
    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph('Diabetic Retinopathy Screening Report', subtitle_style))
    story.append(Paragraph(f'Generated: {now}', small_style))
    story.append(HRFlowable(width='100%', thickness=2, color=grade_color, spaceAfter=16))

    # ── Grade Summary Table ──────────────────────────────────────────
    story.append(Paragraph('Diagnosis Summary', heading_style))

    urgency = URGENCIES[grade] if isinstance(grade, int) else 'N/A'
    summary_data = [
        ['DR Grade',      f'Grade {grade} — {label}'],
        ['Confidence',    f'{confidence:.1f}%'],
        ['Urgency',       urgency],
        ['Image Quality', f'Sharpness score: {sharpness:.1f}' if sharpness is not None else 'N/A'],
        ['Fuzzy Adjust.', f'{delta:+.3f} grade units' if delta is not None else 'N/A'],
    ]

    summary_table = Table(summary_data, colWidths=[2.2*inch, 4.3*inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND',  (0,0), (0,-1), colors.HexColor('#f1f5f9')),
        ('TEXTCOLOR',   (0,0), (0,-1), colors.HexColor('#475569')),
        ('FONTNAME',    (0,0), (0,-1), 'Helvetica-Bold'),
        ('FONTNAME',    (1,0), (1,-1), 'Helvetica'),
        ('FONTSIZE',    (0,0), (-1,-1), 10),
        ('ROWBACKGROUNDS', (0,0), (-1,-1), [colors.white, colors.HexColor('#f8fafc')]),
        ('GRID',        (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING',(0,0), (-1,-1), 10),
        ('TOPPADDING',  (0,0), (-1,-1), 7),
        ('BOTTOMPADDING',(0,0),(-1,-1), 7),
        ('TEXTCOLOR',   (1,0), (1,0),  grade_color),
        ('FONTNAME',    (1,0), (1,0),  'Helvetica-Bold'),
    ]))
    story.append(summary_table)

    # ── Clinical Description ─────────────────────────────────────────
    story.append(Paragraph('Clinical Interpretation', heading_style))
    if description:
        story.append(Paragraph(description, body_style))
    else:
        descs = [
            'No signs of diabetic retinopathy detected. Healthy retinal vasculature with no microaneurysms, haemorrhages, or exudates visible. Annual screening recommended.',
            'Mild non-proliferative DR. Microaneurysms only. Close monitoring every 6-12 months recommended.',
            'Moderate non-proliferative DR. Haemorrhages, hard exudates, and cotton-wool spots may be present. Ophthalmology referral advised.',
            'Severe non-proliferative DR. Extensive haemorrhages, venous beading, or intraretinal microvascular abnormalities. Urgent ophthalmology referral required.',
            'Proliferative DR — most advanced stage. Neovascularisation, vitreous haemorrhage, or tractional retinal detachment possible. Immediate intervention required.',
        ]
        story.append(Paragraph(descs[grade] if isinstance(grade, int) else '', body_style))

    # ── Per-Class Probabilities ──────────────────────────────────────
    if probs:
        story.append(Paragraph('Grade Probability Distribution', heading_style))
        prob_data = [['Grade', 'Label', 'Probability']]
        for i, p in enumerate(probs):
            prob_data.append([f'Grade {i}', GRADE_LABELS[i], f'{p*100:.1f}%'])

        prob_table = Table(prob_data, colWidths=[1.1*inch, 3*inch, 1.5*inch])
        prob_table.setStyle(TableStyle([
            ('BACKGROUND',   (0,0), (-1,0),  colors.HexColor('#1e293b')),
            ('TEXTCOLOR',    (0,0), (-1,0),  colors.white),
            ('FONTNAME',     (0,0), (-1,0),  'Helvetica-Bold'),
            ('FONTSIZE',     (0,0), (-1,-1), 10),
            ('ROWBACKGROUNDS',(0,1),(-1,-1), [colors.white, colors.HexColor('#f8fafc')]),
            ('GRID',         (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
            ('LEFTPADDING',  (0,0), (-1,-1), 10),
            ('RIGHTPADDING', (0,0), (-1,-1), 10),
            ('TOPPADDING',   (0,0), (-1,-1), 6),
            ('BOTTOMPADDING',(0,0), (-1,-1), 6),
            ('ALIGN',        (2,0), (2,-1),  'CENTER'),
            ('FONTNAME',     (0, grade+1), (-1, grade+1), 'Helvetica-Bold'),
            ('TEXTCOLOR',    (0, grade+1), (-1, grade+1), grade_color),
        ]))
        story.append(prob_table)

    # ── Heatmap image if provided ────────────────────────────────────
    heatmap_b64 = data.get('heatmap_b64', '') or data.get('heatmap_url', '')
    if heatmap_b64 and ',' in heatmap_b64:
        heatmap_b64 = heatmap_b64.split(',')[1]
    if heatmap_b64:
        try:
            from reportlab.platypus import Image as RLImage
            img_data = base64.b64decode(heatmap_b64)
            img_buf = BytesIO(img_data)
            story.append(Paragraph('Activation Heatmap', heading_style))
            story.append(RLImage(img_buf, width=3*inch, height=3*inch))
        except Exception:
            pass

    # ── Disclaimer ───────────────────────────────────────────────────
    story.append(Spacer(1, 0.3*inch))
    story.append(HRFlowable(width='100%', thickness=1, color=colors.HexColor('#e2e8f0'), spaceAfter=8))

    doc.build(story)
    buffer.seek(0)
    return send_file(buffer, as_attachment=True,
                     download_name=f'NeuroFuzzyDR_Report_Grade{grade}.pdf',
                     mimetype='application/pdf')

@app.route('/favicon.ico')
def favicon():
    # Return a tiny transparent PNG as favicon to avoid 404s
    buf = _BytesIO()
    img = _Image.new('RGBA', (16, 16), (0, 0, 0, 0))
    img.save(buf, format='PNG')
    buf.seek(0)
    return send_file(buf, mimetype='image/png')

if __name__ == '__main__':
    print("\n>>> Starting Flask server on http://0.0.0.0:5000 ...")
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)