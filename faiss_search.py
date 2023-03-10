import faiss
import json
import numpy as np

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
            count += 1
            element = np.array (self.embeddings[uuid])
            element_matrix = np.vstack((element, element_matrix))
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


    def decode_search_result(self, result): #test with multiple element vectors; gives a vector of positions THAT WAS A GOOD IDEA GWYNNO!
        decoded_results = []
        print ("result shape")
        print (result)
        for position in result:#position is a np ndarray with one thing in [position]
            print ("Position is")
            print (position.type)
            result_vector = self.element_matrix[position]
            for uuid, vector in self.embeddings.items(): #vector is a list, result_vector is an array
                # print (len(vector))
                # print (len(result_vector.tolist()))
                # break
                if result_vector.tolist() == vector: #if this works, might need to add a for x in resilt_vector or something because it might return multiple results
                    decoded_results.append (uuid)
                    #break # avoids duplicates in the images; do I want to do this
        return decoded_results

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
    # for uuid in embeddings:
    #     element2 = np.array(embeddings[uuid]).astype('float32')
    
    #query = np.vstack((element, element2))
    result = srch.search(element, 1)
    final = srch.decode_search_result(result)
    
    print (len(final))

# from PIL import Image
 
# searched = Image.open(f"images/{uuidx}")
# result = Image.open(f"images/{final[5]}")
 
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