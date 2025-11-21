import re
import csv
import json
import random
from tqdm import tqdm
from word2number import w2n
from num2words import num2words
from collections import defaultdict

random.seed(42)

def find_index_case_insensitive(obj, obj_interest):
    # Use list comprehension to create a case-insensitive search
    possible_index = next((i for i, item in enumerate(obj) if item.lower() == obj_interest.lower()))
    return possible_index

def replace_whole_word(sentence, target, replacement):
    return re.sub(r'\b' + re.escape(target) + r'\b', replacement, sentence)

def get_unique_length(lst):
    return 1 if len(set(item[0] for item in lst)) == 1 else len(set(item[0] for item in lst))

def arrange_sent(S, add_index):
    if add_index:
        return "\n".join(f"{i + 1}: {sentence}" for i, sentence in enumerate(S))
    else:
        return "\n\n".join(S)

def get_two_unique_tuples(lst):
    word_map = defaultdict(list)
    [word_map[item[0]].append(item) for item in lst]  # Group tuples by word
    #print(word_map)

    word1, word2 = random.sample(word_map.keys(), 2)
    return [random.choice(word_map[word1]), random.choice(word_map[word2])]
    
def remove_duplicates(input_string):
    lines = input_string.strip().split('\n')
    unique_lines = '\n\n'.join(sorted(set(lines), key=lines.index))  # Remove duplicates, preserving order
    return unique_lines

def convert_to_list(data):
    # Remove leading "-", strip whitespace, and split by either comma or newline
    items = re.split(r",|\n", data)
    return [item.strip().lstrip('-').strip() for item in items if item.strip()]

def extract_entities(S, Index):
    if all(key in S for key in ["<agents>:", "<medical entities>:", "<locations>:", 
                            "Selected Agents:", "Selected Medical Entities:", "Selected Locations:", 
                            "Generated Conversation:"]):
        
        task_1, task_2, task_3 = re.split(r"Task \d Output:", S)[1:]
        task_1_agent = [item.lower() for item in convert_to_list(re.search(r"<agents>:\s*(.*?)(?=<medical entities>)", task_1, re.DOTALL).group(1))]
        task_1_object = [item.lower() for item in convert_to_list(re.search(r"<medical entities>:\s*(.*?)(?=<locations>)", task_1, re.DOTALL).group(1))]
        task_1_location = [item.lower() for item in convert_to_list(re.search(r"<locations>:\s*(.*)", task_1, re.DOTALL).group(1))]

        task_2_selected_agents = [item.lower() for item in convert_to_list(re.search(r"Selected Agents:\s*(.*?)(?=Selected Medical Entities:)", task_2, re.DOTALL).group(1))]
        task_2_selected_objects = [item.lower() for item in convert_to_list(re.search(r"Selected Medical Entities:\s*(.*?)(?=Selected Locations:)", task_2, re.DOTALL).group(1))]
        task_2_selected_locations = [item.lower() for item in convert_to_list(re.search(r"Selected Locations:\s*(.*?)(?=First Triplet)", task_2, re.DOTALL).group(1))]

        task_3_generated_conv = re.search(r"Generated Conversation:\s*(.*)", task_3, re.DOTALL).group(1).strip()
        utterances_task3 = re.findall(r"(A: .*?|B: .*?)(?=\n\s*|\n\n|$)", task_3_generated_conv, re.DOTALL)
        utterance_list_task3 = [utterance.strip().lower() for utterance in utterances_task3]
        
        result = {
            "Seed Conversation Index": Index,
            "Task 1 Agent": task_1_agent,
            "Task 1 Medical Entities": task_1_object,
            "Task 1 Location": task_1_location,
            "Task 2 Selected Agents": task_2_selected_agents,
            "Task 2 Selected Medical Entities": task_2_selected_objects,
            "Task 2 Selected Locations": task_2_selected_locations,
            "Task 3 Generated Conversations": utterance_list_task3
        }

        return result
    else:
        print("Not all required elements are present in the string.")
        return None
    
def Find(words, sentences):
    result = []
    for word in words:
        for i, sentence in enumerate(sentences):
            # Check for the word in its original format (case-insensitive)
            match = re.search(r'\b(' + re.escape(word) + r'|' + re.escape(word.replace('_', ' ')) + r')\b', sentence, re.IGNORECASE)
            if match:
                # Append the matched word exactly as it appears in the sentence
                result.append((match.group(), True, i))
                
    return result

def Replace(tuple_external, tuple_emphasis, sentences):
    index = tuple_emphasis[2]
    old_word = tuple_emphasis[0]
    new_word = tuple_external[0]
    sentence = sentences[index]
    updated_sentence = replace_whole_word(sentence, old_word, new_word)
    sentences[index] = updated_sentence
    return sentences

def Swap(tuple_external, tuple_emphasis, sentences):
    index1 = tuple_external[2]
    index2 = tuple_emphasis[2]
    word1 = tuple_external[0]
    word2 = tuple_emphasis[0]
    sentence1 = sentences[index1]
    sentence2 = sentences[index2]
    
    assert word1 in sentence1 and word2 in sentence2
    
    sentence1 = replace_whole_word(sentence1, word1, word2)
    sentence2 = replace_whole_word(sentence2, word2, word1)
    sentences[index1] = sentence1
    sentences[index2] = sentence2
    return sentences

def perturb_variable_swap(result, Type):
    '''
    Variable Swap: Swap the values of two variables.
    Current Functionality: It only swaps the variables of the same "Type".
    Can be extended to swap the variables between different "Types".
    '''
    new_result = {}
    entity_total, entity_emphasis = [], []
    
    new_result['Seed Conversation Index'] = result['Seed Conversation Index']
    new_result['Task 3 Generated Conversations'] = arrange_sent(result['Task 3 Generated Conversations'], True)
    sentences = result['Task 3 Generated Conversations']
    
    if Type == "agent":
        entity_total, entity_emphasis = result['Task 1 Agent'], result['Task 2 Selected Agents']
    elif Type == "medicalentity":
        entity_total, entity_emphasis = result['Task 1 Medical Entities'], result['Task 2 Selected Medical Entities']
    elif Type == "location":
        entity_total, entity_emphasis = result['Task 1 Location'], result['Task 2 Selected Locations']
        
    if len(set(entity_total)) > len(set(entity_emphasis)):
        entity_external = list(set(entity_total) - set(entity_emphasis))
    else:
        entity_external = list(set(entity_emphasis) - set(entity_total))
    
    # find if any element of the entity_external is in the conversation or not
    entity_external_present = Find(entity_external, result['Task 3 Generated Conversations'])
    
    # collect all occurences of entity_emphasis from the conversation
    entity_emphasis_present = Find(entity_emphasis, result['Task 3 Generated Conversations'])
    
    # external entity present and has atleast two elements
    if get_unique_length(entity_external_present) > 1:
        # randomly select a word from entity_external to swap with a random word from entity_external only
        random_entity_ext_elems = get_two_unique_tuples(entity_external_present)
        updated_conv = Swap(random_entity_ext_elems[0], random_entity_ext_elems[1], sentences)
        new_result['Perturbed conversation'] = updated_conv
        new_result['Utternace Index'] = "swapped between indexes " + str(random_entity_ext_elems[1][2] + 1) + " and " + str(random_entity_ext_elems[0][2] + 1)
        new_result['Old Word'] = random_entity_ext_elems[1][0]
        new_result['New Word'] = random_entity_ext_elems[0][0]
        new_result['Perturbation Type'] = "Variable Swap: " + Type
        new_result['Perturbation Nature'] = "external word swaped by external word"
        new_result['Answer Affected'] = "Not Possible"
        
    # external entity present but only has one element
    elif get_unique_length(entity_external_present) == 1:
        # randomly select a word from entity_external to replace a random word from entity_present
        random_entity_external = random.choice(entity_external_present)
        random_entity_emphasis = random.choice(entity_emphasis_present)
        updated_conv = Swap(random_entity_external, random_entity_emphasis, sentences)
        new_result['Perturbed conversation'] = updated_conv
        new_result['Utternace Index'] = "swapped between indexes " + str(random_entity_emphasis[2] + 1) + "and" + str(random_entity_external[2] + 1)
        new_result['Old Word'] = random_entity_emphasis[0]
        new_result['New Word'] = random_entity_external[0]
        new_result['Perturbation Type'] = "Variable Swap: " + Type
        new_result['Perturbation Nature'] = "emphasis word swaped by external word"
        new_result['Answer Affected'] = "Possible"
        
    # external entity not present
    elif get_unique_length(entity_external_present) == 0 and get_unique_length(entity_emphasis_present) > 1:
        random_entity_emp_elems = get_two_unique_tuples(entity_emphasis_present)
        updated_conv = Swap(random_entity_emp_elems[0], random_entity_emp_elems[1], sentences)
        new_result['Perturbed conversation'] = updated_conv
        new_result['Utternace Index'] = "swapped between indexes " + str(random_entity_emp_elems[1][2] + 1) + " and " + str(random_entity_emp_elems[0][2] + 1)
        new_result['Old Word'] = random_entity_emp_elems[1][0]
        new_result['New Word'] = random_entity_emp_elems[0][0]
        new_result['Perturbation Type'] = "Variable Swap: " + Type
        new_result['Perturbation Nature'] = "emphasis word swaped by emphasis word"
        new_result['Answer Affected'] = "Possible"
        
    else:
        new_result = {}
        
    return new_result
    
def perturb_variable_substitution(result, Type):
    '''
    Variable Substitution: Replace a variable with another one that occurs in the context.
    '''
    new_result = {}
    entity_total, entity_emphasis = [], []
    
    new_result['Seed Conversation Index'] = result['Seed Conversation Index']
    new_result['Task 3 Generated Conversations'] = arrange_sent(result['Task 3 Generated Conversations'], True)
    sentences = result['Task 3 Generated Conversations']
    
    if Type == "agent":
        entity_total, entity_emphasis = result['Task 1 Agent'], result['Task 2 Selected Agents']
    elif Type == "medicalentity":
        entity_total, entity_emphasis = result['Task 1 Medical Entities'], result['Task 2 Selected Medical Entities']
    elif Type == "location":
        entity_total, entity_emphasis = result['Task 1 Location'], result['Task 2 Selected Locations']
        
    if len(set(entity_total)) > len(set(entity_emphasis)):
        entity_external = list(set(entity_total) - set(entity_emphasis))
    else:
        entity_external = list(set(entity_emphasis) - set(entity_total))
    
    # find if any element of the entity_external is in the conversation or not
    entity_external_present = Find(entity_external, result['Task 3 Generated Conversations'])
    
    # collect all occurences of entity_emphasis from the conversation
    entity_emphasis_present = Find(entity_emphasis, result['Task 3 Generated Conversations'])
    
    # external entity present and has atleast two elements
    if get_unique_length(entity_external_present) > 1:
        # randomly select a word from entity_external to replace a random word from entity_external only
        random_entity_ext_elems = get_two_unique_tuples(entity_external_present)
        updated_conv = Replace(random_entity_ext_elems[0], random_entity_ext_elems[1], sentences)
        new_result['Perturbed conversation'] = updated_conv
        new_result['Utternace Index'] = random_entity_ext_elems[1][2] + 1
        new_result['Old Word'] = random_entity_ext_elems[1][0]
        new_result['New Word'] = random_entity_ext_elems[0][0]
        new_result['Perturbation Type'] = "Variable Substitution: " + Type
        new_result['Perturbation Nature'] = "external word replaced by external word"
        new_result['Answer Affected'] = "Not Possible"
        
    # external entity present but only has one element
    elif get_unique_length(entity_external_present) == 1:
        # randomly select a word from entity_external to replace a random word from entity_present
        random_entity_external = random.choice(entity_external_present)
        try:
            random_entity_emphasis = random.choice(entity_emphasis_present)
        except Exception as e:
            print(sentences)
            print(Type)
            print(entity_total, entity_emphasis, entity_external)
            print(entity_external_present, entity_emphasis_present)
            raise IndexError("Cannot choose from an empty sequence")
        updated_conv = Replace(random_entity_external, random_entity_emphasis, sentences)
        new_result['Perturbed conversation'] = updated_conv
        new_result['Utternace Index'] = random_entity_emphasis[2] + 1
        new_result['Old Word'] = random_entity_emphasis[0]
        new_result['New Word'] = random_entity_external[0]
        new_result['Perturbation Type'] = "Variable Substitution: " + Type
        new_result['Perturbation Nature'] = "emphasis word replaced by external word"
        new_result['Answer Affected'] = "Possible"
        
    # external entity not present
    elif get_unique_length(entity_external_present) == 0 and get_unique_length(entity_emphasis_present) > 1:
        random_entity_emp_elems = get_two_unique_tuples(entity_emphasis_present)
        updated_conv = Replace(random_entity_emp_elems[0], random_entity_emp_elems[1], sentences)
        new_result['Perturbed conversation'] = updated_conv
        new_result['Utternace Index'] = random_entity_emp_elems[1][2] + 1
        new_result['Old Word'] = random_entity_emp_elems[1][0]
        new_result['New Word'] = random_entity_emp_elems[0][0]
        new_result['Perturbation Type'] = "Variable Substitution: " + Type
        new_result['Perturbation Nature'] = "emphasis word replaced by emphasis word"
        new_result['Answer Affected'] = "Possible"
        
    else:
        new_result = {}
        
    return new_result

def perturb_quantity_change(result, Type):
    
    '''
    Quantity Change: Replacing a word/number indicating quantity with some other quantity.
    Currently, it makes changes to only one quantity in the entire perturbation. Can be extended to make mulitple changes.
    '''
    
    new_result = {}
    new_result['Seed Conversation Index'] = result['Seed Conversation Index']
    new_result['Task 3 Generated Conversations'] = arrange_sent(result['Task 3 Generated Conversations'], True)
    sentences = result['Task 3 Generated Conversations']
    entity_total, entity_emphasis = result['Task 1 Medical Entities'], result['Task 2 Selected Medical Entities']
    if len(set(entity_total)) > len(set(entity_emphasis)):
        entity_external = list(set(entity_total) - set(entity_emphasis))
    else:
        entity_external = list(set(entity_emphasis) - set(entity_total))
    
    quantity_detect = []
    
    for i in range(len(result['Task 3 Generated Conversations'])):
        temp_sent = result['Task 3 Generated Conversations'][i].split()
        for word in temp_sent:
            if word != "point":
                try:
                    number = w2n.word_to_num(word)
                    quantity_detect.append((i, word, number))
                except ValueError:
                    continue
    
    if len(quantity_detect) > 0:
        random_elem = random.choice(quantity_detect)
        
        if random_elem[2] == 1:
            operator = "add"
        else:
            operator = random.choice(["add", "sub"])
        
        if operator == "add":
            temp_no = random.randint(1, random_elem[2])
            new_no = random_elem[2] + temp_no
        elif operator == "sub":
            try:
                temp_no = random.randint(1, random_elem[2] - 1)
            except Exception as e:
                print(random_elem[2], quantity_detect)
                raise ValueError("empty range for randrange() (1, 0, -1)")
            new_no = random_elem[2] - temp_no
        
        sentence = sentences[random_elem[0]]
        sentence = replace_whole_word(sentence, random_elem[1], num2words(new_no))
        sentences[random_elem[0]] = sentence
        new_result['Perturbed conversation'] = sentences
        new_result['Utternace Index'] = random_elem[0] + 1
        new_result['Old Word'] = random_elem[1]
        new_result['New Word'] = num2words(new_no)
        new_result['Perturbation Type'] = f"Quantity Change {operator} for: " + Type
        
        
        if (any(word in sentence for word in entity_external) or (random_elem[0] > 0 and any(word in sentences[random_elem[0] - 1] for word in entity_external))):
            new_result['Perturbation Nature'] = "one quantity replaced by another quantity for external word"
            new_result['Answer Affected'] = "Not Possible"
        else:
            new_result['Perturbation Nature'] = "one quantity replaced by another quantity for emphasis word"
            new_result['Answer Affected'] = "Possible"
    
    else:
        new_result = {}
            
    return new_result

def perturb_quantifier_change(result):
    
    new_result = {}
    predefined_qu_dict = {"all": "some", "some": "all", "All": "Some", "Some": "All", "ALL": "SOME", "SOME": "ALL"}
    new_result['Seed Conversation Index'] = result['Seed Conversation Index']
    new_result['Task 3 Generated Conversations'] = arrange_sent(result['Task 3 Generated Conversations'], True)
    sentences = result['Task 3 Generated Conversations']
    qu_present = Find(list(predefined_qu_dict.keys()), sentences)
    
    if get_unique_length(qu_present) > 0:
        random_qu = random.choice(qu_present)
        sentence = sentences[random_qu[2]]
        word = random_qu[0]
        sentence = replace_whole_word(sentence, word, predefined_qu_dict[word])
        sentences[random_qu[2]] = sentence
        new_result['Perturbed conversation'] = sentences
        new_result['Utternace Index'] = random_qu[2] + 1
        new_result['Old Word'] = word
        new_result['New Word'] = predefined_qu_dict[word]
        new_result['Perturbation Type'] = "Quantifier Change"
        new_result['Perturbation Nature'] = "one quantifier changed to another."
        new_result['Answer Affected'] = "Possible"
        
    else:
        new_result = {}
    
    return new_result

def perturb_log_conn_change(result):
    
    new_result = {}
    predefined_logconn_dict = {"and": "or", "or": "and", "And": "Or", "Or": "And", "OR": "AND", "AND": "OR"}
    new_result['Seed Conversation Index'] = result['Seed Conversation Index']
    new_result['Task 3 Generated Conversations'] = arrange_sent(result['Task 3 Generated Conversations'], True)
    sentences = result['Task 3 Generated Conversations']
    logconn_present = Find(list(predefined_logconn_dict.keys()), sentences)
    
    if get_unique_length(logconn_present) > 0:
        random_logconn = random.choice(logconn_present)
        sentence = sentences[random_logconn[2]]
        word = random_logconn[0]
        sentence = replace_whole_word(sentence, word, predefined_logconn_dict[word])
        sentences[random_logconn[2]] = sentence
        new_result['Perturbed conversation'] = sentences
        new_result['Utternace Index'] = random_logconn[2] + 1
        new_result['Old Word'] = word
        new_result['New Word'] = predefined_logconn_dict[word]
        new_result['Perturbation Type'] = " Logical Connective Change"
        new_result['Perturbation Nature'] = "one logical connective changed to another."
        new_result['Answer Affected'] = "Possible"
        
    else:
        new_result = {}
    
    return new_result

def format_questions(n_conv, ag, obj, loc):
    is_counter_quant_question = False
    #print(obj, n_conv)
    utter_obj = Find(obj, n_conv)
    
    Final_list_Question = []
    
    quantity_detect = []
    for elem in utter_obj:
        curr_obj = elem[0]
        try:
            curr_utter = n_conv[elem[2] + 1].split()
        except:
            curr_utter = n_conv[elem[2]].split()
            
        for word in curr_utter:
            word = word.split(".")[0]
            try:
                number = w2n.word_to_num(word)
                quantity_detect.append((word, number, curr_obj))
            except ValueError:
                continue
    
    #print("\n*********\n")
    #print(utter_obj, quantity_detect, obj)
    if len(quantity_detect) > 0:
        
        for item in quantity_detect:
            obj_interest = item[2]
            quantity_word_interest = item[0]
            quantity_num_interest = item[1]
            
            #possible_index = obj.index(obj_interest)
            try:
                possible_index = find_index_case_insensitive(obj, obj_interest)
            except:
                continue
                
            try:
                ag_interest = ag[possible_index]
            except:
                ag_interest = ag[0]
                
            try:
                loc_interest = loc[possible_index]
            except:
                loc_interest = loc[0]
            
            question = "Did " + ag_interest + " place " + quantity_word_interest + " " + obj_interest + " at " + loc_interest + "?\n\n" + "Did " + ag_interest + " place " + num2words(quantity_num_interest + 1)  + " " + obj_interest + " at " + loc_interest + "?\n\n" + "Did " + ag_interest + " place " + "all the " + obj_interest + " at " + loc_interest + "?\n\n" + "Did " + ag_interest + " place " + "some of the " + obj_interest + " at " + loc_interest + "?".strip()
            Final_list_Question.append(question)
            is_counter_quant_question = True
            
    else:
        l1 = ag * 2 if len(ag) == 1 else ag
        l2 = obj * 2 if len(obj) == 1 else obj
        l3 = loc * 2 if len(loc) == 1 else loc
        
        question = "Did " + l1[0] + " place " + "some of the " + l2[0] + " at " + l3[0] + "?\n\n" + "Did " + l1[0] + " place " + "all the " + l2[0] + " at " + l3[0] + "?\n\n" + "Did " + l1[1] + " place " + "some of the " + l2[1] + " at " + l3[1] + "?\n\n" + "Did " + l1[1] + " place " + "all the " + l2[1] + " at " + l3[1] + "?".strip()
        Final_list_Question.append(question)
    
    temp_Counter = 0
    if is_counter_quant_question:
        temp_Counter = 1
    
    
    new_final_question_str = arrange_sent(Final_list_Question, False)
    new_final_question_str = remove_duplicates(new_final_question_str)
    return  new_final_question_str, temp_Counter

if __name__ == "__main__":
    
    path = "./store_dir/"
    #conv_f = [json.loads(lines) for lines in open(path + "cicero_conv.json", "r")]
    with open(path + "cicero_conv.json", "r", encoding="utf-8") as file:
        conv_f = json.load(file)
    
    perturb_type = ["Variable Substitution", "Variable Swap", "Quantity Change", "Quantifier Change", "Logical Connective Change"]
    perturb_target = ["agent", "medicalentity", "location"]
    count, counter_type_ques = 0, 0
    for i in tqdm(range(len(conv_f))):
        
        for choosen_perturb_type in perturb_type:
            curr_str = conv_f[i]
            result = extract_entities(curr_str, i)
        
            if choosen_perturb_type == "Variable Substitution":
                choosen_perturb_target = random.choice(perturb_target)
                new_result = perturb_variable_substitution(result, choosen_perturb_target)

            elif choosen_perturb_type == "Variable Swap":
                choosen_perturb_target = random.choice(perturb_target)
                new_result = perturb_variable_swap(result, choosen_perturb_target)

            elif choosen_perturb_type == "Quantity Change":
                choosen_perturb_target = perturb_target[1]
                new_result = perturb_quantity_change(result, choosen_perturb_target)

            elif choosen_perturb_type == "Quantifier Change":
                new_result = perturb_quantifier_change(result)

            elif choosen_perturb_type == "Logical Connective Change":
                new_result = perturb_log_conn_change(result)


            if new_result:
                count += 1
                
                temp_variable_conv = result['Task 3 Generated Conversations']
                new_result['Perturbed conversation'] = arrange_sent(new_result['Perturbed conversation'], True).strip()
                new_result['Emphasis Entities'] = "Agents: " + " ".join(result['Task 2 Selected Agents']) + "\n" + "Medical Entities: " +  " ".join(result['Task 2 Selected Medical Entities']) + "\n" + "Locations: " + " ".join(result['Task 2 Selected Locations'])
                
                #QU, TC = format_questions(temp_variable_conv, result['Task 2 Selected Agents'], result['Task 2 Selected Objects'], result['Task 2 Selected Locations'])
                #new_result['Questions'] = QU
                #counter_type_ques += TC
                
                with open(path + "cicero_perturbed.csv", mode='a', newline='', encoding='utf-8') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames = new_result.keys())

                    if csvfile.tell() == 0:
                        writer.writeheader()

                    writer.writerow(new_result)
            
            else:
                pass
                    
    print(count, counter_type_ques)
            
