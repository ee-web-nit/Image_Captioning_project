from image_captioning.components.text_vectorizer import TextVectorizer

captions = [

    "<start> a dog is running <end>",

    "<start> a man rides bicycle <end>",

]

vectorizer = TextVectorizer()

vectorizer.adapt(captions)

print(vectorizer.vocabulary()[:20])

print(

    vectorizer.vectorize(captions)

)