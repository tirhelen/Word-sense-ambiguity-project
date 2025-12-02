import 

if __name__ == "__main__":
    corpus_dir = "./Corpus-spell-AP88"
    corpus = load_corpus(corpus_dir)
    synthetic_corpus = create_synthetic_corpus(corpus, "car", "speech")
    seeds = pick_seeds(synthetic_corpus,"car","speech")
    instances = extract_instances(synthetic_corpus, "carspeech")

    seeds, decision_list = bootstrap(
        seeds,
        instances,
        iterations=10,
        threshold=2.0
    )
    print(f"\nFinal seed count: {len(seeds)}")
    print("\nTop 10 rules in final decision list:")
    for rule in decision_list[:10]:
        print(rule)

    # 6. Use final decision list to label all instances
    labels = label_corpus(instances, decision_list)
    print("\nSample of final labels:")
    print(labels[:20])