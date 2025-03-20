import queue
class Node:
    def __init__(self, count=None, left_child=None, right_child=None, parent = None,symbol = None):
        self.left_child = left_child
        self.right_child = right_child
        self.parent = parent
        self.count = count
        self.symbol = symbol


def rle_encoding(text: bytes):
    n = len(text)
    compressed_text = bytearray(b'')
    counter =1
    prev_symbol = text[0]
    flag = False
    for i in range(1,n):
        if prev_symbol == text[i]:
            if flag == True:
                compressed_text.append(b"#"[0])
                flag = False
            counter += 1
        else:
            if counter == 1:
                if flag == False:
                    compressed_text.append(b"#"[0])
                    flag = True
                compressed_text.append(prev_symbol)
            else:
                compressed_text.append(counter)
                compressed_text.append(prev_symbol)
                counter = 1
            prev_symbol = text[i]

    if flag == True:
        compressed_text.append(prev_symbol)
        compressed_text.append(b"#"[0])
    else:
        compressed_text.append(counter)
        compressed_text.append(prev_symbol)
    return compressed_text
def rle_decoding(text):
    n = len(text)
    decompressed_text=bytearray(b'')
    i = 0
    while (i<n):
        if (text[i] == 35):
            while(i+1<n and text[i+1]!= 35): # - 35 = "#"
                decompressed_text.append(text[i+1])
                i +=1
            i+=2
        else:
            for j in range(text[i]):
                decompressed_text.append(text[i + 1])
            i +=2

    return decompressed_text
def BWT(text):
    n = len(text)
    BWMatrix = [text[i:] + text[0:i] for i in range(n)]
    BWMatrix.sort()
    last_column = bytearray(b'')
    for i in range(n):
        last_column.append(BWMatrix[i][-1])
    #last_column = "".join([BWMatrix[i][-1] for i in range(n)])
    text_index = BWMatrix.index(text)
    return last_column, text_index
def BWT_decoding(last_column, text_index):
     n = len(last_column)
     BWMatrix = [bytearray(b"") for _ in range(n)]
     for i in range(n):
         for j in range(n):
             BWMatrix[j] = last_column[j:j+1] + BWMatrix[j]
         BWMatrix.sort()
     text = BWMatrix[text_index]
     return text
def better_BWT_decoding(last_column, text_index):
    n = len(last_column)
    P_inverse = counting_sort(last_column)
    text = ""
    j = text_index
    for _ in range(n):
        j = P_inverse[j]
        S = text + last_column[j]
    return S
def counting_sort(S):
    n = len(S)
    m = 128
    T = [0 for _ in range(m)]
    T_sub = T
    for s in S:
        T[ord(s)] += 1
    for j in range(1, m):
        T_sub[j] = T_sub[j - 1] + T[j - 1]
    P = [-1 for _ in range(n)]
    P_inverse = [-1 for _ in range(n)]
    for i in range(n):
        P_inverse[T_sub[ord(S[i])]] = i
        P[i] = T_sub[ord(S[i])]
        T_sub[ord(S[i])] += 1
    return P_inverse
def MTF(text):
     n = len(text)
     L = [chr(i) for i in range(128)]
     res = bytearray(b"")
     for symb in range(n):
         res.append((L.index(chr(text[symb]))))
         i = L.index(chr(text[symb]))
         L = [L[i]] + L[:i] + L[i+1:]
     return res
def MTF_decoding(text):
    n = len(text)
    L = [chr(i) for i in range(128)]
    res = bytearray(b"")
    for symb in range(n):
        i = text[symb]
        res.append(ord(L[i]))
        L = [L[i]] + L[:i] + L[i+1:]
    return res
def codes(root):
    codes={}
    node = root
    current_code = ""
    if node.symbol is not None:
        codes[node.symbol] = current_code
    else:
        codes(node.left_child, current_code + "0")
        codes(node.right_child, current_code + "1")
    return codes
def Haffman_alg(text):
    n = len(text)
    P = [0 for _ in range(n)]
    for symb in text:
        P[ord(symb)]+=1
    P.sort()
    q = queue.PriorityQueue()
    for s in text:
        node = Node(P[ord(s)], None, None, None,s )
        q.put(node)
    while(not q.empty()):
        q1 =q.get()
        if not q.empty():
            q2 = q.get()
            new_node = (q1.count + q2.count, q1, q2)
            q1.parent = new_node
            q2.parent = new_node
            q.put(new_node)

        else:
            root = q1
        code = codes(root)
        return code


file_input = open("test.txt","r",encoding="utf-8")
file_output = open("compressed_test.txt","w",encoding="utf-8")
text = file_input.read()
text = b'banana'
#text = text.encode('utf-8')

#compressed_text = rle_encoding(text)
#decompressed_text = rle_decoding(compressed_text)
last_column, text_index = BWT(text)
print(text_index, last_column)
print(BWT_decoding(last_column, text_index))

#print(compressed_text)
#print(decompressed_text)
#print(MTF(text))
#print(MTF_decoding(MTF(text)))
file_input.close()