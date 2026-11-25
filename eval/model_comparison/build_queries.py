"""
Builds eval/model_comparison/queries.json: 40 health needs, each asked in English,
Hinglish (Roman Hindi) and Hindi (Devanagari), plus 20 unanswerable questions.

Relevance is decided from passage TITLES only (disease pattern + question intent), fixed
before any retriever is run, so no model influences its own labels:
  grade 2 = right disease and right intent,  grade 1 = right disease, other intent.

Limitations: queries were drafted by an AI assistant and need review by Hindi speakers;
a passage that answers the question under an unrelated title counts as not relevant.
"""
import json
import re
from pathlib import Path

INTENTS = {
    "symptoms": r"symptom|sign",
    "treatment": r"treatment|what to do|remed|relief|manage|how to treat|overcome",
    "causes": r"cause",
    "prevention": r"prevent|reduce|manage",
    "diagnosis": r"diagnos",
    "risk": r"at risk|the risk",
    "overview": r"what is \(are\)|information about",
}

# (id, title pattern, exclude pattern, intent, English, Hinglish, Hindi)
T = [
 ("diabetes_sym", r"\b(pre)?diabetes\b", r"insipidus|syndrome|neonatal|anomal|deafness", "symptoms",
  "how can I tell if my blood sugar is too high", "sugar ki bimari ke lakshan kya hote hain", "शुगर की बीमारी के लक्षण क्या होते हैं"),
 ("diabetes_prev", r"\b(pre)?diabetes\b", r"insipidus|syndrome|neonatal|anomal|deafness", "prevention",
  "ways to avoid becoming diabetic", "diabetes se kaise bache", "मधुमेह से कैसे बचें"),
 ("glaucoma_risk", r"glaucoma", r"syndrome|sleep apnea", "risk",
  "who is more likely to get raised eye pressure disease", "kala motia ka khatra kise zyada hota hai", "काला मोतिया का खतरा किसे ज्यादा होता है"),
 ("asthma_treat", r"asthma", r"syndrome", "treatment",
  "what helps with wheezing and breathing trouble in asthma", "dama ka ilaj kya hai", "दमा का इलाज क्या है"),
 ("dengue_sym", r"dengue", r"^$", "symptoms",
  "signs of dengue after a mosquito bite", "dengue bukhar ke lakshan batao", "डेंगू बुखार के लक्षण बताओ"),
 ("malaria_sym", r"\bmalaria\b", r"^$", "symptoms",
  "how do you know you have malaria", "malaria hone par kya hota hai", "मलेरिया होने पर क्या होता है"),
 ("anemia_cause", r"^[^?]*\b(iron-deficiency )?anemia\b", r"fanconi|diamond|sideroblast|hemolytic|aplastic|pernicious|dyserythro|megaloblast|thiamine|sickle|syndrome|congenital|x-linked", "causes",
  "why does someone not have enough red blood", "khoon ki kami kyun hoti hai", "खून की कमी क्यों होती है"),
 ("fever_treat", r"what is \(are\) fever|viral fever", r"^$", "treatment",
  "how to bring down a high temperature", "tez bukhar kaise utare", "तेज बुखार कैसे उतारें"),
 ("stroke_sym", r"\bstroke\b", r"syndrome|heat stroke|melas", "symptoms",
  "warning signs that the brain has lost blood supply", "lakwa aane se pehle ke sanket", "लकवा आने से पहले के संकेत"),
 ("kidneystone_prev", r"kidney stone", r"^$", "prevention",
  "how to stop stones forming in the kidneys again", "gurde ki pathri se bachne ka tarika", "गुर्दे की पथरी से बचने का तरीका"),
 ("bp_treat", r"high blood pressure|hypertension", r"pulmonary|portal|intracranial|ocular|syndrome|brachydactyly|pheochromo", "treatment",
  "how to lower my BP", "BP high rehta hai kya karu", "बीपी हाई रहता है क्या करूँ"),
 ("heartattack_sym", r"heart attack|sudden chest pain", r"^$", "symptoms",
  "what does a heart attack feel like", "dil ka daura padne ke lakshan", "दिल का दौरा पड़ने के लक्षण"),
 ("epilepsy_treat", r"^[^?]*\b(epilepsy|seizures?)\b", r"northern|juvenile|pyridoxine|benign|familial|autosomal|syndrome|progressive|rolandic|childhood|infancy|infantile|lateral|occipital|reading|myoclonic", "treatment",
  "medicine for repeated fits", "mirgi ke daure ka ilaj", "मिर्गी के दौरे का इलाज"),
 ("psoriasis_treat", r"\bpsoriasis\b", r"generalized pustular", "treatment",
  "what can be done for thick scaly skin patches from psoriasis", "psoriasis chamdi ki bimari ka ilaj", "सोरायसिस त्वचा रोग का इलाज"),
 ("gout_cause", r"\bgout\b", r"syndrome", "causes",
  "why does uric acid make my big toe swell", "uric acid badhne se jodo me sujan kyun hoti hai", "यूरिक एसिड बढ़ने से जोड़ों में सूजन क्यों होती है"),
 ("migraine_treat", r"\bmigraine\b", r"hemiplegic|basilar|syndrome|abdominal", "treatment",
  "how to ease throbbing pain on one side of the head", "aadhe sar ka dard kaise theek kare", "आधे सिर का दर्द कैसे ठीक करें"),
 ("tb_diag", r"tuberculosis", r"^$", "diagnosis",
  "tests to find out if I have TB", "TB ki jaanch kaise hoti hai", "टीबी की जांच कैसे होती है"),
 ("pneumonia_sym", r"^[^?]*\bpneumonia\b", r"cryptogenic|desquamative|interstitial|eosinophilic", "symptoms",
  "how does a lung infection like pneumonia show itself", "nimoniya ke lakshan kya hain", "निमोनिया के लक्षण क्या हैं"),
 ("jointpain_cause", r"osteoarthritis|rheumatoid arthritis|what is \(are\) arthritis|joint pain", r"juvenile|syndrome|inherited", "causes",
  "why do my knees and fingers hurt", "ghutno aur ungliyo me dard kyun hota hai", "घुटनों और उंगलियों में दर्द क्यों होता है"),
 ("hypothyroid_sym", r"hypothyroidism", r"congenital|syndrome|alopecia|resistance", "symptoms",
  "signs of an underactive thyroid", "thyroid kam hone ke lakshan", "थायराइड कम होने के लक्षण"),
 ("gallstone_treat", r"gallstone", r"^$", "treatment",
  "how are stones in the gallbladder removed", "pitte ki pathri ka ilaj", "पित्त की पथरी का इलाज"),
 ("hepb_prev", r"hepatitis b\b", r"^$", "prevention",
  "how to avoid catching hepatitis B", "hepatitis B se bachav kaise kare", "हेपेटाइटिस बी से बचाव कैसे करें"),
 ("constipation_treat", r"constipation", r"^$", "treatment",
  "I cannot pass stool easily, what should I do", "kabz ho gayi hai kya karna chahiye", "कब्ज हो गई है क्या करना चाहिए"),
 ("diarrhea_treat", r"\bdiarrhea\b", r"chloride|congenital|syndrome|bile acid|polyendocrin", "treatment",
  "how to stop loose motions", "dast lag gaye hai kaise rokein", "दस्त लग गए हैं कैसे रोकें"),
 ("insomnia_treat", r"insomnia", r"fatal", "treatment",
  "I lie awake all night, how can I sleep better", "raat ko neend nahi aati kya karu", "रात को नींद नहीं आती क्या करूँ"),
 ("depression_sym", r"\bdepression\b", r"syndrome", "symptoms",
  "how to know if constant sadness is an illness", "depression ke lakshan kya hote hain", "अवसाद के लक्षण क्या होते हैं"),
 ("cataract_treat", r"^[^?]*\bcataracts?\b", r"cerulean|hyperferr|hutterite|congenital|syndrome|microcornea|autosomal|anterior|posterior|total|ataxia|juvenile|x-linked|deafness", "treatment",
  "how is a cloudy eye lens fixed", "motiyabind ka operation aur ilaj", "मोतियाबिंद का ऑपरेशन और इलाज"),
 ("jaundice_ov", r"\bjaundice\b", r"^$", "overview",
  "why do eyes and skin turn yellow", "piliya kya hota hai", "पीलिया क्या होता है"),
 ("acne_ov", r"what is \(are\) acne", r"^$", "overview",
  "pimples on my face", "chehre par muhase kyun nikalte hain", "चेहरे पर मुंहासे क्यों निकलते हैं"),
 ("obesity_cause", r"\bobesity\b", r"hypoventilation|syndrome|retinal|inherited|melanocortin|leptin", "causes",
  "reasons people gain too much weight", "motapa kyun badhta hai", "मोटापा क्यों बढ़ता है"),
 ("cholesterol_prev", r"cholesterol", r"hypercholesterol|cholesteryl|syndrome", "prevention",
  "foods that bring down fat in the blood", "cholesterol kam karne ke liye kya khaye", "कोलेस्ट्रॉल कम करने के लिए क्या खाएं"),
 ("parkinson_sym", r"parkinson'?s? disease", r"juvenile", "symptoms",
  "hand shaking and slow movement in old age disease signs", "parkinson me haath kaapne ke lakshan", "पार्किंसन में हाथ कांपने के लक्षण"),
 ("alzheimer_risk", r"alzheimer", r"caregiv", "risk",
  "what makes memory loss disease more likely", "bhulne ki bimari alzheimer ka khatra kisko hai", "भूलने की बीमारी अल्जाइमर का खतरा किसको है"),
 ("chickenpox_ov", r"chickenpox", r"^$", "overview",
  "itchy blister rash that children catch", "chhoti mata kya hoti hai", "छोटी माता क्या होती है"),
 ("backpain_treat", r"back pain", r"^$", "treatment",
  "my lower back aches, how do I get relief", "kamar dard ka upay", "कमर दर्द का उपाय"),
 ("cough_treat", r"^[^?]*\bcough\b", r"whooping", "treatment",
  "how to get rid of a dry throat tickle", "sukhi khansi ka ilaj", "सूखी खांसी का इलाज"),
 ("peptic_ov", r"peptic ulcer|stomach ache|gastritis|indigestion|heartburn|acidity", r"syndrome", "overview",
  "burning pain in stomach after eating", "khane ke baad pet me jalan hoti hai", "खाने के बाद पेट में जलन होती है"),
 ("hemorrhoid_treat", r"hemorrhoid", r"^$", "treatment",
  "swollen veins near the anus, how to treat", "bawasir ka ilaj", "बवासीर का इलाज"),
 ("rabies_sym", r"\brabies\b", r"^$", "symptoms",
  "what happens after a mad dog bite infection", "pagal kutte ke kaatne se hone wali bimari ke lakshan", "पागल कुत्ते के काटने से होने वाली बीमारी के लक्षण"),
 ("osteoporosis_prev", r"osteoporosis", r"juvenile|pseudoglioma|syndrome", "prevention",
  "how to keep bones from becoming weak", "haddiyo ko kamzor hone se kaise bachaye", "हड्डियों को कमजोर होने से कैसे बचाएं"),
]

# Off-topic and "health-like but fictional" questions: nothing in the corpus answers these.
UNANSWERABLE = [
 "what is the capital of France", "how to write a binary search in Python", "best tourist spots in Tokyo",
 "how to play guitar chords", "speed of light in vacuum", "how to fix a leaking sink", "who won the cricket world cup in 2011",
 "price of gold today", "how to renew my passport", "which laptop is good for gaming",
 "bharat ke pradhan mantri kaun hain", "dilli se mumbai ki train kab hai", "sabse acchi bike kaun si hai",
 "biryani kaise banate hain", "aaj ka mausam kaisa hai",
 "भारत की राजधानी क्या है", "ताजमहल किसने बनवाया", "मोबाइल रिचार्ज कैसे करें", "क्रिकेट के नियम क्या हैं", "पासपोर्ट कैसे बनवाएं",
]


# Rare genetic conditions that merely contain a common disease name, plus inheritance/epidemiology pages.
RARE = re.compile(r"genetic changes|inherited|how many people|mitochondrial|short stature|dysplasia|intellectual|alopecia"
                  r"|bronchiolitis|hydrocephalus|pyridoxal|hypochromic|iron-refractory|tetralogy|nanophthalmos|retinopathy"
                  r"|dandy|cardiomyopathy|microcephaly|primary infantile|hereditary|research", re.I)


def grade(title: str, topic) -> int:
    t = title.lower()
    _, pat, exc, intent, *_ = topic
    if RARE.search(t) or not re.search(pat, t) or re.search(exc, t):
        return 0
    return 2 if re.search(INTENTS[intent], t) else 1


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from backend.ir.inverted_index import InvertedIndex
    chunks = list(InvertedIndex.load().chunks_map.values())
    titles = sorted({c["question"] for c in chunks})
    out = {"answerable": [], "unanswerable": UNANSWERABLE}
    for t in T:
        g = {x: grade(x, t) for x in titles}
        g = {k: v for k, v in g.items() if v}
        assert g, f"no relevant titles for {t[0]}"
        out["answerable"].append({"id": t[0], "intent": t[3], "en": t[4], "hinglish": t[5], "hindi": t[6], "relevant_titles": g})
        print(f"{t[0]}: {len(g)} titles, {sum(v == 2 for v in g.values())} grade-2 | " + "; ".join(list(g)[:14]))
    p = Path(__file__).with_name("queries.json")
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("saved", p)
