from fastcoref import FCoref, spacy_component
import spacy
def run_coref():

    # Load the coreference resolution model
    predictor = Predictor.from_path(
        "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz"
    )

    # Input text
    text = "John went to the market. He bought some apples. John loves to eat them."

    # Perform coreference resolution
    resolved_output = predictor.predict(document=text)

    # Access the coreference clusters
    clusters = resolved_output["clusters"]
    print("Coreference clusters:", clusters)

    # Access the coreference resolved text
    resolved_text = resolved_output["document"]
    print("Resolved text:", resolved_text)

def coreference(dict):

    model = FCoref()

    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("fastcoref")

    for key in dict:
        corefenced_sentences = []
        for item in dict[key]:
            doc = nlp(item)
            doc._.coref_clusters
            doc = nlp(
                item,
                component_cfg={"fastcoref": {'resolve_text': True}}
            )
            corefenced_sentences.append(doc._.resolved_text)
            dict[key] = corefenced_sentences

    return dict

