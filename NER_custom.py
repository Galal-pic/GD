import spacy
import random
from spacy import util
from spacy.tokens import Doc
from spacy.training import Example
from spacy.language import Language

def print_doc_entities(_doc: Doc):
    if _doc.ents:
        for _ent in _doc.ents:
            print(f"     {_ent.text} {_ent.label_}")
    else:
        print("     NONE")


def customizing_pipeline_component(nlp: Language):
    # NOTE: Starting from Spacy 3.0, training via Python API was changed. For information see - https://spacy.io/usage/v3#migrating-training-python
    train_data = [
        (f'I need to learn python to be data analyst.', [(16, 22, 'skill')])
    ]

    # Result before training
    print(f"\nResult BEFORE training:")
    doc = nlp(u'I need to learn python to be data analyst.')
    print_doc_entities(doc)

    # Disable all pipe components except 'ner'
    disabled_pipes = []
    for pipe_name in nlp.pipe_names:
        if pipe_name != 'ner':
            nlp.disable_pipes(pipe_name)
            disabled_pipes.append(pipe_name)

    print("   Training ...")
    optimizer = nlp.create_optimizer()
    for _ in range(25):
        random.shuffle(train_data)
        for raw_text, entity_offsets in train_data:
            doc = nlp.make_doc(raw_text)
            example = Example.from_dict(doc, {"entities": entity_offsets})
            nlp.update([example], sgd=optimizer)

    # Enable all previously disabled pipe components
    for pipe_name in disabled_pipes:
        nlp.enable_pipe(pipe_name)

    # Result after training
    print(f"Result AFTER training:")
    doc = nlp(u'I need to learn python to be data analyst.')
    print_doc_entities(doc)

def main():
    nlp = spacy.load('en_core_web_sm')
    customizing_pipeline_component(nlp)


if __name__ == '__main__':
    main()