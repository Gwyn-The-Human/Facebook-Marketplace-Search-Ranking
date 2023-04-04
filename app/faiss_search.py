import faiss
import json
import numpy as np


class FaissSearch():
    """
The search takes an array of embeddings (can be a single embedding in the array), and returns a list of lists
with the k specified nearest matches for each embedding in the array. 
    """

    def __init__(self):
        """
        Loads the embeddings from a JSON file, builds a matrix of each elements UUID, and builds
        an faiss index of that matrix. 
        """
        with open("image_embeddings.json", 'r') as f:
            self.embeddings = json.load(f)
        self.element_matrix = self.build_element_matrix()
        self.index = self.build_index() #so in theory I can just return self.index[index]
        

    def build_element_matrix(self):
        """
        Builds a matrix of each elements UUID. 

        Returns:
            A matrix of the UUIDS the elements in the corpus.
        """
        count = 0 # for faster testing
        element_matrix = np.ndarray([0,1000]) # don't have to hard code this; could put embeddings.shape[1]
        for uuid in self.embeddings:
            element = np.array (self.embeddings[uuid])
            element_matrix = np.vstack((element, element_matrix))
            count += 1
            if count == 100: #for faster testing
                break
        return element_matrix.astype('float32')


    def build_index(self):
        """
        Creates a flat index of the element matrix using the faiss module.
        
        Returns:
            The index is being returned.
        """
        index = faiss.IndexFlatL2(1000)  
        index.add(self.element_matrix)
        return index                  
        

    def search(self, query_vector, num_results): 
        """
        Returns the distances and indices of the closest vectors to the queried vectors. 
        The query vector can contain the features of one or several images. A set of results 
        will be returned for each set of image features in the quesry vector. 
        
        Args:
            query_vector: a vector containing one or more vectors you want to search for
            num_results: number of results to return

        Returns:
            The index of the closest vector to the query vector.
        """
        d, i = self.index.search(query_vector, num_results)
        return i


    def decode_search_result(self, result): 
        decoded_results = [] # a list of lists of UUIDS of near images for each queried element
        for queried_element in result: # result is an array of arrays, each which is the collection of nearest images for each element queried. 
            uuids = [] #creates a list of UUIDS for each queried element, to be appended to the total decoded results
            for position in queried_element:#position is a np ndarray with one element in [position]
                result_vector = self.element_matrix[position]
                for uuid, vector in self.embeddings.items(): #vector is a list, result_vector is an array
                    if result_vector.tolist() == vector[0]: 
                        uuids.append (uuid)
            decoded_results.append(uuids)
        return decoded_results # a list of lists 


if __name__ == '__main__':
    pass



