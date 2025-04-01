from train import load_and_tokenize_data
from imblearn.over_sampling import SMOTE
import numpy as np
from collections import defaultdict
from datasets import Dataset

dataset, label_names = load_and_tokenize_data()
# print(label_names)

training_dataset = dataset["train"]
sentences = training_dataset["ner_tags"]
counting = defaultdict(int)
arr_of_number_of_entities = []
arr_of_number_of_non_entities = []
for sentence in sentences:
    number_of_entities = 0 
    number_of_non_entities = 0
    for ner_tag in sentence:
        counting[label_names[ner_tag]] += 1
        if label_names[ner_tag] == "O":
            number_of_non_entities += 1
        else:
            number_of_entities += 1
    arr_of_number_of_entities.append(number_of_entities)
    arr_of_number_of_non_entities.append(number_of_non_entities)
# Print the counts in a formatted way
# print("Entity Counts:")
# for label, count in counting.items():
#     print(f"{label}: {count}")

# Count occurrences of each label
# counting = [0] * len(label_names)
# for sentence in sentences:
#     for ner_tag in sentence:
#         counting[ner_tag] += 1

# print(counting)
# print(label_names)
total_entities = sum(count for label, count in counting.items() if label != "O")
# print("Total number of entities:", total_entities)  # Count of all other entities
# print("total number of non-entity", counting[label_names[ner_tag]])  # Count of 'O' (no entity)




# print("arr_of_number_of_entities", (arr_of_number_of_entities))
# print("arr_of_number_of_non_entities", (arr_of_number_of_non_entities))

import matplotlib.pyplot as plt

# Plot a histogram for arr_of_number_of_entities
plt.figure(figsize=(10, 6))
plt.hist(arr_of_number_of_entities, bins=range(min(arr_of_number_of_entities), max(arr_of_number_of_entities) + 2), color='blue', alpha=0.7, edgecolor='black')
plt.title("Frequency of Number of Entities per Sentence")
plt.xlabel("Number of Entities")
plt.ylabel("Frequency")
plt.show()

# Plot a histogram for arr_of_number_of_non_entities
plt.figure(figsize=(10, 6))
plt.hist(arr_of_number_of_non_entities, bins=range(min(arr_of_number_of_non_entities), max(arr_of_number_of_non_entities) + 2), color='green', alpha=0.7, edgecolor='black')
plt.title("Frequency of Number of Non-Entities per Sentence")
plt.xlabel("Number of Non-Entities")
plt.ylabel("Frequency")
plt.show()



def draw_pie_chart():
    # label, size = [], []
    # total = sum(y for x, y in counting.items())
    total = total_entities+counting[label_names[ner_tag]]
    # for x, y in counting.items():
    #     label.append(x)
    #     size.append(y/total*100)
    size = [total_entities/total, counting[label_names[ner_tag]]/total]
    label = ['Entity', 'Non-entity']
    color = ['lightblue', 'red']
    plt.pie(size, explode=(0.1, 0), labels=label, colors=color, autopct='%1.1f%%')
    plt.show()

# Test
# draw_pie_chart()

def filter_dataset(k, dataset: Dataset):
    filtered = dataset.filter(lambda data: len(data["ner_tags"])-data["ner_tags"].count(0) >= k)
    return filtered

# Test 

# print(training_dataset["ner_tags"])


def find_sentence_ratio(sentence: list):
    entity, non_entity = 0, 0
    for x in sentence:
        if x == 0:
            non_entity += 1
        else:
            entity += 1
    return entity/(entity+non_entity)

def find_dataset_mean_ratio(sentences): 
    
    arr_of_ratios = []
    for sentence in sentences:
        entity_ratio = find_sentence_ratio(sentence)    
        arr_of_ratios.append(entity_ratio)
    return np.mean(arr_of_ratios) 


# filtered_dataset = filter_dataset(16, dataset)
# print(filtered_dataset["ner_tags"]) 


def plot_pie_chart(k):
    filtered_dataset = filter_dataset(k, training_dataset)
    filtered_sentences = filtered_dataset["ner_tags"]
    
    counting = defaultdict(int)
    for sentence in filtered_sentences:
        for ner_tag in sentence:
            counting[label_names[ner_tag]] += 1
    
    total_entities = sum(count for label, count in counting.items() if label != "O")
    total_non_entities = counting["O"]
    total = total_entities + total_non_entities
    
    size = [total_entities / total, total_non_entities / total]
    label = ['Entity', 'Non-entity']
    color = ['lightblue', 'red']
    
    plt.figure(figsize=(8, 8))
    plt.pie(size, explode=(0.1, 0), labels=label, colors=color, autopct='%1.1f%%', startangle=140)
    plt.title(f"Entity vs Non-Entity Distribution for k={k}")
    plt.show()

for k in range(0, 25):
    plot_pie_chart(k)

#haha