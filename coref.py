import stanza


def run_coref():

    # Download the English models if not already downloaded
    stanza.download('en')

    # Initialize the pipeline
    nlp = stanza.Pipeline('en')

    # Define the sentence
    sentence = "Diyon is a boy. He is awesome."

    # Process the sentence
    doc = nlp(sentence)

    print(doc.sentences)

    for sent in doc.sentences:

        for word in sent.words:

            if word.upos == "PRON":

                antecedent = word.deprel
                print(word, antecedent)

    # # Iterate over the sentences
    # for sent in doc.sentences:
    #     # Iterate over the words in the sentence
    #     for word in sent.words:
    #         print(sent)
    #         # Check if the word is a pronoun
    #         if word.upos == 'PRON':
    #             # Find the antecedent by traversing the dependency tree
    #             antecedent = word.head
    #             while antecedent.deprel != 'root':
    #                 antecedent = sent.words[antecedent.head - 1]
    #             # Print the resolved pronoun with its antecedent
    #             print(f"Pronoun: {word.text}\tAntecedent: {antecedent.text}")


