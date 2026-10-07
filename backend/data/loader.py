"""
backend/data/loader.py
Dataset Loader Module for HealthNest Q&A System.
Implements flexible CSV parsing, schema auto-detection, null row filtering,
and duplicate detection using Python standard library 'csv'.
"""

import csv
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def detect_columns(header: List[str]) -> Dict[str, str]:
    """
    Auto-detect column names from CSV header using mapping in config.py.
    Returns dict mapping standard field names -> actual CSV column header names.
    """
    header_clean = [h.strip() for h in header]
    header_lower_map = {h.lower(): h for h in header_clean}
    
    mapping = {}
    for std_field, candidates in config.CSV_COLUMN_MAPPING.items():
        matched = None
        for candidate in candidates:
            if candidate.lower() in header_lower_map:
                matched = header_lower_map[candidate.lower()]
                break
        if matched:
            mapping[std_field] = matched

    if "question" not in mapping or "answer" not in mapping:
        raise ValueError(
            f"CSV Header missing required 'question' or 'answer' columns! Found: {header_clean}"
        )
    return mapping


def load_dataset(data_dir: Path = config.RAW_DATA_DIR) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Reads all CSVs in the raw data directory using python stdlib 'csv.DictReader'.
    Filters out empty rows and detects duplicates by question string.
    
    Returns:
        (records, stats_summary)
    """
    if not list(data_dir.glob("*.csv")):
        csv_path = config.RAW_CSV_PATH
        logger.warning(f"No CSV files found in {data_dir}. Generating default sample dataset...")
        generate_sample_dataset(csv_path)

    records = []
    stats = {"total_rows_read": 0, "valid_records": 0, "empty_rows_dropped": 0, "duplicates_found": 0}
    seen_questions = set()

    for csv_path in data_dir.glob("*.csv"):
        logger.info(f"Loading {csv_path}...")
        with open(csv_path, mode="r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            col_map = detect_columns(reader.fieldnames or [])
            
            for idx, row in enumerate(reader):
                stats["total_rows_read"] += 1
                question = row.get(col_map.get("question", ""), "").strip()
                answer = row.get(col_map.get("answer", ""), "").strip()

                if not question or not answer:
                    stats["empty_rows_dropped"] += 1
                    continue

                # Deduplication based on exact normalized question text
                q_norm = question.lower()
                if q_norm in seen_questions:
                    stats["duplicates_found"] += 1
                    continue
                seen_questions.add(q_norm)

                qa_id = str(row.get(col_map.get("qa_id", ""), f"qa_{len(records)+1}")).strip()
                category = row.get(col_map.get("category", ""), "General Health").strip() if col_map.get("category") else "General Health"
                source = row.get(col_map.get("source", ""), csv_path.name).strip() if col_map.get("source") else csv_path.name

                record = {
                    "qa_id": qa_id,
                    "question": question,
                    "answer": answer,
                    "category": category,
                    "source": source
                }
                records.append(record)

    stats["valid_records"] = len(records)
    logger.info(f"Loaded dataset stats: {stats}")
    return records, stats


def generate_sample_dataset(target_path: Path) -> None:
    """
    Generates a rich, realistic health Q&A CSV dataset for testing & demonstration
    if no raw Kaggle CSV is present locally.
    """
    target_path.parent.mkdir(parents=True, exist_ok=True)
    
    sample_categories = [
        "General Health", "Medications", "Lifestyle", "Cardiology",
        "Pediatrics", "Dermatology", "Gastroenterology", "Neurology"
    ]
    
    base_templates = [
        ("What are common causes and relief methods for headache and migraine?",
         "Headache relief and causes involve distinguishing between tension headaches, migraines, cluster headaches, and sinus pressure. Tension headaches are frequently triggered by stress, dehydration, eye strain, lack of sleep, or poor cervical posture. Migraines present as pulsating, throbbing pain often accompanied by nausea, aura, and photophobia (sensitivity to light and sound). Effective headache relief methods include resting in a quiet, dark room, applying a cold ice pack to the temples or forehead, drinking plenty of water for rehydration, and gentle neck massage. Over-the-counter pain relievers such as Paracetamol (Acetaminophen) or NSAIDs like Ibuprofen provide rapid symptomatic relief when taken early. Chronic or severe migraines may require prescription triptans (such as Sumatriptan) and avoiding dietary triggers like caffeine withdrawal, aged cheeses, or artificial sweeteners.",
         "Neurology"),
        ("How to get quick relief from severe headache and sir dard?",
         "Immediate relief from severe headache and sir dard involves non-pharmacological interventions combined with targeted analgesic therapy. Rest in a dark, quiet room with minimal sensory stimulation. Apply a cold compress to the forehead or temples to constrict dilated blood vessels, or use a warm compress on the neck muscles if the headache is tension-related. Hydrate immediately with electrolyte-rich water. Over-the-counter analgesics such as Ibuprofen (400 mg) or Paracetamol (500-1000 mg) can alleviate acute pain. Avoid alcohol, nicotine, and excessive screen time. If a headache is sudden, explosive (thunderclap), or accompanied by neck stiffness, high fever, or neurological weakness, seek emergency medical care immediately.",
         "Neurology"),
        ("What are the warning signs and home remedies for tension headaches?",
         "Tension headaches produce a dull, aching band-like pressure across both sides of the forehead and the base of the skull. Common causes include mental stress, prolonged computer or smartphone screen use, fatigue, and jaw clenching. Home remedies include practicing progressive muscle relaxation, taking 15-minute breaks every hour of desk work, applying peppermint oil or a warm compress to the back of the neck, and maintaining adequate hydration. Over-the-counter analgesics like Aspirin or Acetaminophen can be used intermittently. Regular physical exercise and consistent sleep schedules prevent recurrent episodes.",
         "Neurology"),
        ("What are the early symptoms of diabetes?", 
         "Early symptoms of diabetes include increased thirst (polydipsia), frequent urination (polyuria), extreme hunger, weight loss, fatigue, blurred vision, and slow-healing sores. Blood sugar levels should be tested if these occur.",
         "Cardiology"),
        ("How to manage high blood pressure naturally?",
         "High blood pressure (hypertension) can be managed naturally by eating a low-sodium DASH diet, regular aerobic exercise for 30 minutes daily, limiting alcohol consumption, avoiding smoking, managing stress, and maintaining a healthy body mass index (BMI under 25).",
         "Cardiology"),
        ("What is the recommended dosage for Paracetamol in adults?",
         "The standard adult dosage for Paracetamol (Acetaminophen) is 500 mg to 1000 mg every 4 to 6 hours as needed, with a maximum daily limit of 4000 mg (4 grams) in 24 hours. Exceeding this can cause severe liver damage.",
         "Medications"),
        ("What causes sudden chest pain and when to seek emergency help?",
         "Sudden chest pain can be caused by angina, myocardial infarction (heart attack), pericarditis, acid reflux, or muscle strain. If chest pain radiates to the left arm, jaw, or neck accompanied by shortness of breath, seek immediate emergency help (call 112).",
         "Cardiology"),
        ("What are common remedies for severe stomach ache and gastritis?",
         "Stomach ache and gastritis remedies include drinking antacids, consuming ginger tea, avoiding spicy or acidic foods, drinking plenty of fluids, and eating smaller frequent meals. If stomach pain is severe or accompanied by blood in stool, consult a doctor.",
         "Gastroenterology"),
        ("How does Metformin work for Type 2 Diabetes treatment?",
         "Metformin lowers blood glucose levels by suppressing glucose production by the liver (gluconeogenesis) and increasing insulin sensitivity in peripheral tissues. Common side effects include mild nausea and stomach upset.",
         "Medications"),
        ("What are the side effects of Amoxicillin antibiotic?",
         "Amoxicillin is a broad-spectrum penicillin antibiotic. Common side effects include nausea, vomiting, diarrhea, skin rash, and abdominal distress. Seek urgent medical care if severe allergic reactions like swelling or breathing difficulty develop.",
         "Medications"),
        ("How to improve sleep quality and overcome insomnia?",
         "To overcome insomnia and improve sleep quality, establish a consistent sleep schedule, avoid blue light screens 1 hour before bedtime, limit caffeine intake after 2 PM, keep the bedroom cool and dark, and practice deep breathing exercises.",
         "Lifestyle"),
        ("What are the primary causes and treatments for asthma attacks?",
         "Asthma is a chronic inflammatory airway condition triggered by allergens, cold air, exercise, or stress. Inhaled corticosteroids provide long-term control, while short-acting beta-agonists (Albuterol inhalers) act as quick rescue medications during acute bronchospasm.",
         "General Health"),
        ("What causes joint pain in fingers and knees?",
         "Joint pain in fingers and knees is commonly caused by osteoarthritis (wear and tear of cartilage), rheumatoid arthritis (autoimmune inflammation), gout (uric acid accumulation), or tendonitis. Treatment involves NSAIDs, physical therapy, and warm compresses.",
         "General Health"),
        ("How to reduce high cholesterol levels through diet?",
         "To lower LDL cholesterol, increase intake of soluble fiber (oats, beans, apples), eat omega-3 rich foods like salmon and flaxseeds, replace saturated fats with monounsaturated olive oil, and limit trans fats and processed meats.",
         "Lifestyle"),
        ("What are the warning signs of a stroke?",
         "Stroke warning signs follow the FAST acronym: Face drooping, Arm weakness, Speech difficulty, Time to call emergency services. Prompt thrombolytic treatment within 3 hours significantly reduces long-term brain damage.",
         "Neurology"),
        ("How to treat skin rashes and itching at home?",
         "Home treatment for skin rashes includes applying calamine lotion or 1% hydrocortisone cream, taking cool oatmeal baths, using fragrance-free moisturizers, and taking oral antihistamines like Cetirizine for itching relief.",
         "Dermatology"),
        ("What is the normal blood pressure reading for adults?",
         "A normal adult blood pressure reading is less than 120/80 mmHg. Systolic pressure between 120-129 mmHg is elevated, while 130/80 mmHg or higher indicates stage 1 hypertension requiring lifestyle changes or medication.",
         "Cardiology"),
        ("What are the symptoms and home treatment for acidity and heartburn?",
         "Heartburn presents as a burning sensation in the chest caused by stomach acid regurgitating into the esophagus. Home relief includes over-the-counter antacids, avoiding lying down immediately after meals, and elevating the head of your bed.",
         "Gastroenterology"),
        ("What are the best home remedies for viral fever and high body temperature?",
         "Viral fever management includes rest, continuous hydration with water, soups, and ORS solutions, lukewarm water sponge baths to bring down body temperature, and Paracetamol (500-650 mg) for temperature control. Avoid self-medicating with antibiotics since viral infections do not respond to antibacterial drugs. Seek emergency care if temperature exceeds 103°F (39.4°C) or is accompanied by confusion.",
         "General Health"),
        ("What are effective treatments for dry cough and sore throat?",
         "Effective relief for dry cough and sore throat includes warm salt water gargles 3-4 times daily, drinking herbal tea with honey and ginger, using steam inhalation, and taking throat lozenges. Over-the-counter cough suppressants such as Dextromethorphan help reduce nighttime coughing. If a cough persists for more than 3 weeks or involves blood, immediate clinical evaluation is needed.",
         "General Health"),
        ("How to treat common cold, runny nose, and sinus congestion?",
         "Common cold treatment focuses on symptomatic relief: staying well-hydrated, inhaling eucalyptus steam to clear congested nasal passages, using saline nasal sprays, and taking antihistamines like Cetirizine for runny nose and sneezing. Decongestant nasal sprays should not be used for more than 3 consecutive days to avoid rebound congestion.",
         "General Health"),
        ("What are the causes and home remedies for lower back pain?",
         "Lower back pain is commonly caused by lumbar muscle strain, ligament sprain, poor ergonomics, prolonged sitting, or lumbar disc herniation. Immediate home care includes applying ice packs during the first 48 hours followed by heat therapy, gentle pelvic tilts, walking, and short-term use of NSAIDs like Ibuprofen. Avoid prolonged bed rest as gentle movement accelerates recovery.",
         "General Health"),
        ("How to manage acute diarrhea, loose motions, and dehydration?",
         "The primary treatment for acute diarrhea and loose motions is rapid rehydration with Oral Rehydration Salts (ORS) solution to replace lost electrolytes and fluids. Eat bland foods adhering to the BRAT diet (bananas, rice, applesauce, toast), and consider zinc supplementation and probiotics to restore gut flora. Avoid dairy, high-fat foods, and caffeinated beverages until symptoms resolve.",
         "Gastroenterology"),
        ("What causes acute anxiety and panic attacks, and how to calm down?",
         "Panic attacks and acute anxiety are characterized by rapid heartbeat (palpitations), trembling, shortness of breath, dizziness, and intense fear. Calming techniques include the 4-7-8 breathing method (inhale for 4 seconds, hold for 7, exhale for 8), the 5-4-3-2-1 sensory grounding exercise, splashing cold water on the face to activate the mammalian dive reflex, and reminding oneself that the physical sensation is temporary and harmless.",
         "Lifestyle"),
        ("What are the symptoms and prevention tips for kidney stones?",
         "Kidney stones produce sharp, severe cramping pain in the back and flank that radiates to the lower abdomen and groin, often accompanied by pink or cloudy urine and nausea. Prevention relies on drinking at least 2.5 to 3 liters of water daily to dilute stone-forming minerals, reducing dietary sodium, and moderating animal protein and oxalate-rich foods.",
         "General Health"),
        ("How does Cetirizine work for allergic reactions and skin itching?",
         "Cetirizine is a second-generation antihistamine that selectively blocks peripheral H1 histamine receptors, preventing symptoms such as sneezing, runny nose, itchy watery eyes, and urticaria (hives). It is usually taken as a single 10 mg dose daily and has a lower sedative profile compared to first-generation antihistamines like Diphenhydramine.",
         "Medications"),
        ("How to prevent digital eye strain and computer vision syndrome?",
         "Digital eye strain is prevented by following the 20-20-20 rule: every 20 minutes look at an object at least 20 feet away for at least 20 seconds. Ensure appropriate ambient lighting without screen glare, position the monitor about 20 to 28 inches from the eyes slightly below eye level, blink consciously to lubricate the cornea, and use preservative-free artificial tear drops if dryness occurs.",
         "Lifestyle")
    ]
    
    rows = [
        {
            "qa_id": f"qa_{idx:05d}",
            "question": question,
            "answer": answer,
            "category": category,
            "source": "HealthNest Curated Q&A"
        }
        for idx, (question, answer, category) in enumerate(base_templates, start=1)
    ]

    with open(target_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["qa_id", "question", "answer", "category", "source"])
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Generated sample dataset with {len(rows)} diverse Q&A records at {target_path}")
