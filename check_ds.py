from datasets import load_dataset

FIM_TOKEN_DICT = {'google/codegemma-7b': {'prefix': '<|fim_prefix|>', 'middle': '<|fim_middle|>', 'suffix': '<|fim_suffix|>'},
                  'meta-llama/CodeLlama-7b-hf' :{'prefix': '<PRE>', 'middle': '<MID>', 'suffix': '<SUF>'},
                  'Qwen/CodeQwen1.5-7B' :{'prefix': '<fim_prefix>', 'middle': '<fim_middle>', 'suffix': '<fim_suffix>'},
                  'bigcode/starcoder2-7b' :{'prefix': '<fim_prefix>', 'middle': '<fim_middle>', 'suffix': '<fim_suffix>'},
                  'ibm-granite/granite-8b-code-base' :{'prefix': '<fim_prefix>', 'middle': '<fim_middle>', 'suffix': '<fim_suffix>'}}



def check_ds():
    langs = ["Chinese", "Dutch", "English", "Polish", 'Greek']
    for l in langs:
        ds = load_dataset("AISE-TUDelft/multilingual-code-comments", l, split='train')

        for rec in ds:
            #Check for any nan fields
            for key in rec.keys():
                #not for error_codes_<llm> fields
                if key.startswith('error_codes_'):
                    continue
                
                if rec[key] is None:
                    print(f"Found None in lang {l}, key {key}, record id {rec['file_id']}")
                
            # Check for empty strings
            for key in rec.keys():
                if isinstance(rec[key], str) and rec[key].strip() == "":
                    print(f"Found empty string in lang {l}, key {key}, record id {rec['file_id']}")

            # Check that FIM tokens are present in the predict_<llm> fields
            for llm in FIM_TOKEN_DICT.keys():
                pred_field = f"predict_{llm}"
                if pred_field in rec:
                    fim_tokens = FIM_TOKEN_DICT[llm]
                    if not (fim_tokens['prefix'] in rec[pred_field] and
                            fim_tokens['middle'] in rec[pred_field] and
                            fim_tokens['suffix'] in rec[pred_field]):
                        print(f"Missing FIM tokens in lang {l}, llm {llm}, record id {rec['file_id']}")


if __name__ == "__main__":
    check_ds()