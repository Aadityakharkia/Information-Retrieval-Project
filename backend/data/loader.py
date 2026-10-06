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


def load_dataset(csv_path: Path = config.RAW_CSV_PATH) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Reads CSV using python stdlib 'csv.DictReader'.
    Filters out empty rows and detects duplicates by question string.
    
    Returns:
        (records, stats_summary)
    """
    if not csv_path.exists():
        logger.warning(f"CSV file not found at {csv_path}. Generating default sample dataset...")
        generate_sample_dataset(csv_path)

    records = []
    stats = {"total_rows_read": 0, "valid_records": 0, "empty_rows_dropped": 0, "duplicates_found": 0}
    seen_questions = set()

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

            qa_id = str(row.get(col_map.get("qa_id", ""), f"qa_{idx+1}")).strip()
            category = row.get(col_map.get("category", ""), "General Health").strip() if col_map.get("category") else "General Health"
            source = row.get(col_map.get("source", ""), "Medical Q&A Archive").strip() if col_map.get("source") else "Medical Q&A Archive"

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


def generate_sample_dataset(target_path: Path, num_records: int = 500) -> None:
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
         "Gastroenterology")
    ]
    
    rows = []
    for i in range(1, num_records + 1):
        tpl = base_templates[(i - 1) % len(base_templates)]
        q_var = f"{tpl[0]}" if i <= len(base_templates) else f"{tpl[0]} (Case #{i})"
        a_var = f"{tpl[1]} Reference document #{i} for medical inquiry."
        rows.append({
            "qa_id": f"qa_{i:05d}",
            "question": q_var,
            "answer": a_var,
            "category": tpl[2],
            "source": f"Medical Journal #{100 + (i % 50)}"
        })

    with open(target_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["qa_id", "question", "answer", "category", "source"])
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Generated sample dataset with {num_records} Q&A records at {target_path}")
