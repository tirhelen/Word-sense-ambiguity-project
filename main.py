from WSD import read_file, tokenize, load_corpus, create_synthetic_corpus, pick_seeds, collect_collocations
from WSD import count_features, create_decision_list, extract_instances, label_instance, label_corpus, bootstrap
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np


def get_correct_labels(instances, word1, word2):
    """
    Extract correct labels based on ORIGINAL=word1 / ORIGINAL=word2 markers
    inside left/right context.
    Returns list of labels (0, 1, or None)
    """

    label1 = f"ORIGINAL={word1}"
    label2 = f"ORIGINAL={word2}"

    correct = []
    # Go through every instance of amibiguous word
    for left, target, right in instances:
        # Check if the instance contains "ORIGINAL=word1" nearby
        if label1 in left or label1 in right:
            correct.append(0)
        # Check if the instance contains "ORIGINAL=word2" nearby
        elif label2 in left or label2 in right:
            correct.append(1)
        else:
            correct.append(None)

    return correct


def evaluate_predictions(predictions, correct_labels, word1, word2):
    """
    Evaluates predictions with accuracy, confusion matrix 
    and classification report.
    """
    filtered_predictions = []
    filtered_corrects = []

    for predicted, correct in zip(predictions, correct_labels):
        if predicted is not None:      # Filter out None predictions
            filtered_predictions.append(predicted)
            filtered_corrects.append(correct)

    # Nothing to evaluate if all predictions were None
    if len(filtered_predictions) == 0:
        return

    # Accuracy
    accuracy = np.mean(np.array(filtered_predictions) == np.array(filtered_corrects))
    print(f"Accuracy: {accuracy:.4f}")

    # Confusion matrix
    cm = confusion_matrix(filtered_corrects, filtered_predictions, labels=[0,1])
    print("\nConfusion matrix (rows = true, cols = predicted):")
    print(cm)

    # Precision, recall, F1
    print("\nClassification report:")
    print(classification_report(filtered_corrects, filtered_predictions, target_names=[f"{word1} (0)", f"{word2} (1)"]))

if __name__ == "__main__":

    word1 = "love"
    word2 = "hate"
    combined = word1 + word2
    corpus_dir = "./Corpus-spell-AP88"
    corpus = load_corpus(corpus_dir)
    synthetic_corpus = create_synthetic_corpus(corpus, word1, word2)
    seeds = pick_seeds(synthetic_corpus,word1,word2)
    instances = extract_instances(synthetic_corpus, combined)

    seeds, decision_list = bootstrap(
        seeds,
        instances,
        iterations=10,
        threshold=2.0
    )
    #print(f"\nFinal seed count: {len(seeds)}")
    #print("\nTop 10 rules in final decision list:")
    #for rule in decision_list[:10]:
    #    print(rule)

    # Final step for the WSD: Use final decision list to label all instances
    labels = label_corpus(instances, decision_list)
    #print("\nSample of final labels:")
    #print(labels[:20])

    # Evaluation
    predictions = [sense for sense, llr in labels]
    corrects = get_correct_labels(instances, word1, word2)
    evaluate_predictions(predictions, corrects, word1, word2)