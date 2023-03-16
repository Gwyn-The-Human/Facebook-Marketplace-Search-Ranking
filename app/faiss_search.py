import faiss
import json
import numpy as np


#the search takes an array of embeddings(can be a single one in the array), and returns a list of lists
# with the k specified nearest matches for each embedding in the array. 
class FaissSearch():
    def __init__(self):
        with open("image_embeddings.json", 'r') as f:
            self.embeddings = json.load(f)
        self.element_matrix = self.build_element_matrix()
        self.index = self.build_index()


    def build_element_matrix(self):
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
        index = faiss.IndexFlatL2(1000)  
        index.add(self.element_matrix)
        return index                  
        

    def search(self, query_vector, num_results): # query can include multiple elements!! 
        d, i = self.index.search(query_vector, num_results)
        return i


    def decode_search_result(self, result): 
        decoded_results = [] # a list of lists of UUIDS of near images for each queried element
        for queried_element in result: # result is an array of arrays, each which is the collection of nearest images for each element queried. 
            uuids = [] #creates a list of UUIDS for each queried element, to be appended to the total decoded results
            for position in queried_element:#position is a np ndarray with one thing in [position]
                result_vector = self.element_matrix[position]
                for uuid, vector in self.embeddings.items(): #vector is a list, result_vector is an array
                    if result_vector.tolist() == vector[0]: #if this works, might need to add a for x in resilt_vector or something because it might return multiple results
                        uuids.append (uuid)
            decoded_results.append(uuids)
        return decoded_results # a list of lists 

if __name__ == '__main__':
    



#testing search
    srch = FaissSearch() 
    uuidx = None   
    with open("image_embeddings.json", 'r') as f:
        embeddings = json.load(f)
    for uuid in embeddings:
        element = np.array(embeddings[uuid]).astype('float32')
        uuidx = uuid
        break
    for uuid in embeddings:
         element2 = np.array(embeddings[uuid]).astype('float32')
    
    query = np.vstack((element, element2))
    result = srch.search(element, 2)
    print (element.type) #n
    print (len(query[0]))
    final = srch.decode_search_result(result)
    
    print (final)

# from PIL import Image
# print (uuidx)
# searched = Image.open(f"images/{uuidx}")
# result = Image.open(f"images/{final[0][1]}")
 
# searched.show()
# result.show()








# index is a search space =/= other meaning of index

# vectoir of length 1000 
# and i have n of those 

# thats an n by d matrix 
# build the matrix 

# can then im.show

# when i create an index; it will just calculat the L2 norm of all of those.
# the index is an object that is being built; 