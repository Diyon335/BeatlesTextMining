from fastcoref import FCoref, spacy_component
import spacy


def run_coref():
    model = FCoref()

    preds = model.predict(
        texts=['We are so happy to see you using our coref package. This package is very fast!']
    )

    print(preds[0].get_clusters(as_strings=False))

    print(preds[0].get_clusters())

    print(preds[0].get_logit(
        span_i=(33, 50), span_j=(52, 64)
    ))

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

