"""Comprehensive Inference Test Suite for TriEvo Clinical NER."""

import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from src.inference.pipeline import ClinicalNERPipeline

TEST_CASES = [
    {
        "id": "cardiology_1",
        "description": "Acute Myocardial Infarction and Antiplatelet Therapy",
        "text": "The patient presented with acute myocardial infarction and was immediately treated with aspirin and clopidogrel."
    },
    {
        "id": "metabolic_1",
        "description": "Type 2 Diabetes and Hypertension Polypharmacy",
        "text": "History of type 2 diabetes mellitus and essential hypertension; currently taking metformin 1000mg and lisinopril 20mg."
    },
    {
        "id": "neurology_1",
        "description": "Parkinsonism and Drug-Induced Dyskinesia",
        "text": "The elderly male with Parkinson's disease developed severe dyskinesia following prolonged treatment with levodopa."
    },
    {
        "id": "oncology_1",
        "description": "Chemotherapy-Induced Cardiotoxicity",
        "text": "Doxorubicin therapy was discontinued after the patient showed symptoms of congestive heart failure."
    },
    {
        "id": "complex_chemical_1",
        "description": "Complex Biochemical Compound and Toxic Phenotype",
        "text": "Administration of 1-methyl-4-phenyl-1,2,3,6-tetrahydropyridine produced nigrostriatal lesion in animal models."
    },
    {
        "id": "negative_case_1",
        "description": "Negative Case without Entities",
        "text": "The physical examination was within normal limits and the patient was discharged in stable condition."
    },
    {
        "id": "edge_case_empty",
        "description": "Edge Case - Empty Input",
        "text": ""
    },
    {
        "id": "edge_case_single_drug",
        "description": "Edge Case - Single Chemical Token",
        "text": "Cisplatin"
    },
    {
        "id": "edge_case_single_disease",
        "description": "Edge Case - Single Disease Token",
        "text": "Leukemia"
    }
]


def run_tests(model_path: str):
    print(f"\n=======================================================")
    print(f"Running Inference Verification on Model: {model_path}")
    print(f"=======================================================\n")
    
    pipeline = ClinicalNERPipeline(model_path)
    all_passed = True
    
    for case in TEST_CASES:
        print(f"\nTest [{case['id']}]: {case['description']}")
        print(f"Input: \"{case['text']}\"")
        
        result = pipeline.extract_entities(case["text"])
        entities = result["entities"]
        print(f"Extracted Entities ({len(entities)}):")
        
        for ent in entities:
            # Verification 1: Offset validity
            extracted_sub = case["text"][ent["start"]:ent["end"]]
            offset_valid = (extracted_sub == ent["text"])
            
            # Verification 2: Confidence bounds
            conf = ent["confidence"]
            conf_valid = (0.0 <= conf <= 1.0)
            
            status = "PASS" if (offset_valid and conf_valid) else "FAIL"
            if status == "FAIL":
                all_passed = False
                
            print(f"  - [{ent['label']}] \"{ent['text']}\" (offsets: {ent['start']}:{ent['end']}, conf: {conf:.4f}) [{status}]")
            if not offset_valid:
                print(f"    ERROR: Offset mismatch! Sliced text: '{extracted_sub}' != Entity text: '{ent['text']}'")
                
    print(f"\n=======================================================")
    if all_passed:
        print("ALL INFERENCE TESTS PASSED SUCCESSFULLY!")
    else:
        print("SOME INFERENCE TESTS ENCOUNTERED FAILURES.")
    print(f"=======================================================\n")
    return all_passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=str(root_dir / "models" / "best_clinical_ner"))
    args = parser.parse_args()
    run_tests(args.model_path)
