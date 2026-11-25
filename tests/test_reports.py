"""Medical test reports: verbatim lines, and comparisons only when they are certain."""
import io

import pytest

from backend.memory import report
from backend.memory.prescription import analyse, medicines
from test_patient_memory import ask, client, make_pdf, net  # noqa: F401 (fixtures)

LAB = ["Sunrise Diagnostics", "Patient: Test Person  Age: 40", "COMPLETE BLOOD COUNT", "Test  Result  Unit  Reference Range",
       "Hemoglobin 10.2 g/dL 13.0 - 17.0", "WBC Count 7,800 /cumm 4000 - 11000", "Total Cholesterol 242 mg/dL < 200", "HbA1c 7.4 % 4.0 - 5.6",
       "HBsAg Non Reactive", "Iron Supplement 60 mg advised"]
SCAN = ["City Imaging Centre", "USG WHOLE ABDOMEN", "Liver is mildly enlarged with grade I fatty change.", "Both kidneys are normal in size.",
        "IMPRESSION: Grade I fatty liver."]


@pytest.mark.parametrize("line,status", [
    ("Hemoglobin 10.2 g/dL 13.0 - 17.0", "below"), ("Hemoglobin 13.0 g/dL 13.0-17.0", "within"), ("Hemoglobin 17.1 g/dL 13.0 – 17.0", "above"),
    ("WBC Count 7,800 /cumm 4,000 - 11,000", "within"), ("Platelets 1,50,000 /cumm 1,50,000 - 4,50,000", "within"),
    ("Total Cholesterol 200 mg/dL < 200", "above"), ("Total Cholesterol 199 mg/dL <200", "within"), ("HDL 35 mg/dL > 40", "below"),
    ("SGPT 40 U/L Up to 40", "within"), ("eGFR 60 mL/min/1.73m2 >= 60", "within"), ("Hemoglobin 10.2 L g/dL 13.0 - 17.0", "below"),
    ("Platelet Count 2.1 lakh/cumm 1.5 - 4.5", "within"), ("Creatinine *1.9* mg/dL 0.6 - 1.2", "above"), ("TSH 3.2 uIU/mL (0.4 to 4.2)", "within")])
def test_clear_rows_are_compared(line, status):
    assert report.parse_row(line)["status"] == status


@pytest.mark.parametrize("line", [
    "Hemoglobin 10.2 g/dL M: 13.0 - 17.0 F: 12.0 - 15.0",   # two ranges
    "Hemoglobin 10.2 9.8 g/dL 13.0 - 17.0",                 # two results
    "Hemoglobin 13.0 - 17.0 10.2 g/dL",                     # range printed before the result
    "Hemoglobin g/dL 13.0 - 17.0 10.2",
    "Platelet Count 2.1 lakh/cumm 150000 - 450000",         # different scales
    "Hemoglobin 10.2 H g/dL 13.0 - 17.0",                   # contradicts the lab's own flag
    "Hemoglobin 10.2 g/dL 17.0 - 13.0",                     # impossible range
    "HBsAg Non Reactive", "Glucose 110 mg/dL", "Hemoglobin 10.2 g/dL", "Urea 5 < 10 - 20"])
def test_ambiguous_rows_are_never_compared(line):
    row = report.parse_row(line)
    assert row is None or row["status"] is None


@pytest.mark.parametrize("line", ["Collected on 5 12-03-2026", "Phone 080 2345-6789", "Patient Age 40 18 - 60", "Sample time 9 10:30 - 11:30",
                                  "Tab. Metformin 500 mg 1 - 0 - 1", "Dolo 650 1-0-1 after food", "Page 1 2 - 3"])
def test_non_results_are_not_rows(line):
    assert report.parse_row(line) is None


def test_document_kinds_and_no_fake_medicines():
    assert report.is_report("\n".join(LAB)) and report.is_report("\n".join(SCAN))
    assert not report.is_report("City Clinic\nTab. Metformin 2.5 mg twice daily\nCap Amoxicillin 500 mg 1-0-1")
    assert medicines("\n".join(LAB)) == []


def lines(q, *docs):
    a = analyse(q, [{"id": i, "filename": f"d{i}.pdf", "text": "\n".join(t)} for i, t in enumerate(docs)])
    return [l["text"] for s in a["sources"] for l in s["lines"]], a


@pytest.mark.parametrize("q,expected", [
    ("what is my hemoglobin", [LAB[4]]), ("is my cholesterol high", [LAB[6]]), ("mera sugar kitna hai", [LAB[7]]), ("what is my HbA1c level", [LAB[7]]),
    ("do I have fatty liver", [SCAN[2], SCAN[4]])])
def test_specific_questions_find_the_right_lines(q, expected):
    assert lines(q, LAB, SCAN)[0] == expected


@pytest.mark.parametrize("q", ["what does my blood report say", "meri report me kya hai", "explain my test report"])
def test_whole_report_questions_show_everything(q):
    assert lines(q, LAB)[0] == LAB


@pytest.mark.parametrize("q", ["what are the symptoms of asthma", "how to lower cholesterol", "pet me dard ka ilaj", "what is hemoglobin"])
def test_general_questions_do_not_use_reports(q):
    assert not lines(q, LAB, SCAN)[1]["used"]


def test_abnormal_lists_only_out_of_range_and_says_what_was_not_compared():
    got, a = lines("which of my results are abnormal", LAB)
    assert got == [LAB[4], LAB[6], LAB[7]] and "could not be compared" in a["remarks"][0]


def test_nothing_out_of_range_is_not_reported_as_healthy():
    got, a = lines("are any of my results abnormal", ["Lab", "Hemoglobin 14 g/dL 13 - 17", "TSH 3 uIU/mL 0.4 - 4.2", "HBsAg Reactive"])
    assert got == [] and a["used"] and "2 other line(s) could not be compared" in a["remarks"][0]


def test_missing_test_is_not_answered_with_another_report():
    assert lines("what is my HbA1c level", SCAN)[0] == []
    assert "No line in your reports" in lines("what is my HbA1c in my report", SCAN)[1]["remarks"][0]


def test_prescription_question_ignores_reports():
    rx = ["Dr. Rao", "Tab. Metformin 2.5 mg daily"]
    assert lines("what medicines did my doctor prescribe", LAB, rx)[0] == [rx[1]]


def test_end_to_end_report_stays_local(client, net):  # noqa: F811
    r = client.post("/api/documents", data={"file": (io.BytesIO(make_pdf(LAB)), "lab.pdf")}).get_json()
    assert r["kind"] == "report" and "Hemoglobin" in r["tests"]
    body = ask(client, "why is my cholesterol high", model="openai/gpt-oss-120b")
    first = body["sentences"][0]
    assert first["text"] == LAB[6] and first["origin"] == "report" and first["note"].startswith("Above the range printed on your report")
    assert "database" in {s["origin"] for s in body["sentences"]}
    assert net["get"] == [] and all(u.startswith("http://localhost") for u, _ in net["post"])


@pytest.mark.parametrize("test,topic,ok", [("Cholesterol", "high blood cholesterol", True), ("Cholesterol", "cholesterol", True),
                                           ("Hemoglobin", "hemoglobin e disease", False), ("Hemoglobin", "sickle cell disease", False),
                                           ("Vitamin D", "vitamin d deficiency", True), ("Platelet", "platelet storage pool deficiency", False)])
def test_only_pages_about_the_test_itself(test, topic, ok):
    assert report.page_is_about(test, topic) is ok


def test_low_value_never_pulls_in_a_disease_page(client, net):  # noqa: F811
    client.post("/api/documents", data={"file": (io.BytesIO(make_pdf(LAB)), "lab.pdf")})
    body = ask(client, "explain my test report")
    assert not any("disease" in s["question"].lower() for s in body["sources"])
    assert "Hemoglobin" in body["missing_medicines"]
    assert body["wikipedia_offer"]["query"] == body["missing_medicines"][0]  # the test name only, never the value
    assert net["get"] == []
