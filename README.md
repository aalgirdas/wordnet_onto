# wordnet_onto - WordNet word sense disambiguation (WSD) and WordNet ontology creation

This GitHub page provides all the resources needed for the WSD task. The WordNet semantic dictionary is used to define the semantic meanings of a word. The main feature of the system is that in order to use it, it is enough to download one file from this GitHub page:

principal_components.zip  (18.8 MB).

The fastest way to try the solution presented here is to simply open the Google Colab link and see how this WSD method is integrated into the general framework of NLP tasks in the given notebook.

Colab demo: https://colab.research.google.com/drive/18f_zHHAsLw5wTvwIsqbbCt0vHUo3eXkh#scrollTo=_NksFVBkD8ty

I also recommend watching a YouTube video in which I briefly explain the main aspects of the presented solution.

https://www.youtube.com/watch?v=jO65aQDJ2OM


After unzipping the principal_components.zip file, you get four files, where each file is dedicated to the corresponding grammatical form of the word: noun, verb, adjective and adverb:


principal_components_nou.json
principal_components_ver.json
principal_components_adv.json
principal_components_adj.json


As far as one of these files is concerned, the structure is the same and corresponds to the data structure of the Python dictionary. The key in such a dictionary is the lemma of the word, so when solving the WSD problem, we have to determine the lemma and the grammatical form of the word, and then we extract from this dictionary all the elements needed to solve the WSD problem. These elements are as follows:


pcs – main components;
X_embedded - examples used to train the model. Each example is the corresponding projection of embedded vectors in the principal components;
y – synset key values ​​of X_embedded samples;
k_neighbors_win – a parameter of the kNN model that is recommended for predicting synset values;

How to use the provided model.


Here we will present a code fragment from the Colab document where you will find the complete solution. The following code snippet captures the essence of the method presented.


    principal_components = np.array(components_data['pcs'])
    y_embedded = np.array(components_data['y'])
    X_embedded = np.array(components_data['X_embedded'])
    k_neighbors_win = components_data.get('k_neighbors_win', 1)
    knn = KNeighborsClassifier(n_neighbors=k_neighbors_win)
    knn.fit(X_embedded, y_embedded)
    X_embedded_test = np.dot(X, principal_components.T).reshape(1, -1)
    predicted_rezults = knn.predict(X_embedded_test)



principal_components = np.array(components_data['pcs'])
 y_embedded = np.array(components_data['y'])
 X_embedded = np.array(components_data['X_embedded'])

We take lemma NCA components principal_components, x_embedded coordinate samples used for model training and y_embedded - sample synset key values.

    knn = KNeighborsClassifier(n_neighbors=k_neighbors_win)
    knn.fit(X_embedded, y_embedded)

We are preparing a kNN classifier.

X_embedded_test = np.dot(X, principal_components.T).reshape(1, -1)


We get NCA projections for a specific word in our sentence.

predicted_results = knn.predict(X_embedded_test)

Using the kNN classifier, we predict the synset key value.

The presented method achieves good accuracy results, but its biggest advantage is simplicity and small resources.







