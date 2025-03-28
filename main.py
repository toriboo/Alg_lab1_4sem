import queue
import numpy
from PIL import Image
import time
import struct


class Node():
    def __init__(self, symbol = None, counter = None, left_child = None, right_child =None, parent = None):
        self.symbol = symbol
        self.counter = counter
        self.left = left_child
        self.right = right_child
        self.parent = parent
    def __lt__(self, other):
        return self.counter < other.counter
def new_rle(text):
    n = len(text)
    compressed_text = bytearray(b'')
    counter= 1
    prev_symbol = text[0]
    buffer = bytearray(b'')
    for i in range(1, n):
        if prev_symbol == text[i]:
            if (len(buffer) > 0):
                while (len(buffer) > 127):
                    compressed_text.append(127+128)
                    compressed_text.extend(buffer[:127])
                    buffer = buffer[127:]
                if (len(buffer)> 0):
                    compressed_text.append(len(buffer)+128)
                    compressed_text.extend(buffer)
                    buffer = bytearray(b'')
            counter += 1

        else:
            if (counter > 1):
                while (counter > 127):
                    compressed_text.append(127)
                    compressed_text.append(prev_symbol)
                    counter -= 127
                if (counter > 0):
                    compressed_text.append(counter)
                    compressed_text.append(prev_symbol)
                counter = 1
            else:
                buffer.append(prev_symbol)
        prev_symbol = text[i]

    if (len(buffer) > 0):
        buffer.append(prev_symbol)
        while (len(buffer) > 127):
            compressed_text.append(127 + 128)
            compressed_text.extend(buffer[:127])
            buffer = buffer[127:]
        if (len(buffer) > 0):
            compressed_text.append(len(buffer) + 128)
            compressed_text.extend(buffer)
    else:
        while (counter > 127):
            compressed_text.append(127)
            compressed_text.append(prev_symbol)
            counter -= 127
        if (counter > 0):
            compressed_text.append(counter)
            compressed_text.append(prev_symbol)
    return compressed_text
def new_rle_decoding(text):
    n = len(text)
    i =0
    decompressed_text = bytearray(b'')
    while (i<n):
        if (text[i]< 128):
            for j in range(text[i]):
                if(i+1 < n): decompressed_text.append(text[i + 1])
            i +=2
        else:
            for j in range(text[i]-128):
                if (i + 1 < n): decompressed_text.append(text[i+1])
                i += 1
            i+=1
    return decompressed_text


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
                while (counter > 127):
                    compressed_text.append(127)
                    compressed_text.append(prev_symbol)
                    counter -= 127
                if (counter > 0):
                    compressed_text.append(counter)
                    compressed_text.append(prev_symbol)
                counter = 1
            prev_symbol = text[i]

    if flag == True:
        compressed_text.append(prev_symbol)
        compressed_text.append(b"#"[0])
    else:
        while (counter > 127):
            compressed_text.append(127)
            compressed_text.append(prev_symbol)
            counter -= 127
        if (counter > 0):
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
    text = bytearray(b"")
    j = text_index
    for _ in range(n):
        j = P_inverse[j]
        text.append(last_column[j])
    return text
def counting_sort(S):
    n = len(S)
    m = 256
    T = [0 for _ in range(m)]
    T_sub = [0 for _ in range(m)]
    for s in S:
        T[(s)] += 1
    for j in range(1, m):
        T_sub[j] = T_sub[j - 1] + T[j - 1]
    P = [-1 for _ in range(n)]
    P_inverse = [-1 for _ in range(n)]
    for i in range(n):
        P_inverse[T_sub[(S[i])]] = i
        P[i] = T_sub[(S[i])]
        T_sub[(S[i])] += 1
    return P_inverse
def MTF(text):
     n = len(text)
     L = [chr(i) for i in range(256)]
     res = bytearray(b"")
     for symb in range(n):
         res.append((L.index(chr(text[symb]))))
         i = L.index(chr(text[symb]))
         L = [L[i]] + L[:i] + L[i+1:]
     return res
def MTF_decoding(text):
    n = len(text)
    L = [chr(i) for i in range(256)]
    res = bytearray(b"")
    for symb in range(n):
        i = text[symb]
        res.append(ord(L[i]))
        L = [L[i]] + L[:i] + L[i+1:]
    return res
def count_symb(S):
    N = len(S)
    counter = numpy.array([0 for _ in range(256)])
    for s in S:
        counter[(s)] += 1
    return counter
def prob_estimate(S):
    N = len(S)
    P = numpy.array([0 for _ in range(256)])
    for s in S:
        P[(s)] += 1
    P = P/N
    return P
def entropy(S):
    P = prob_estimate(S)
    P = numpy.array(list(filter(lambda x: x!=0,P)))
    E = -sum(numpy.log2(P) * P)
    return E
def Haffman_alg(text):
    n = len(text)
    Counters_symb  = count_symb(text)
    leafs = []
    q = queue.PriorityQueue()
    for i in range(256):
        if Counters_symb[i] != 0:
            node = Node(symbol=(i), counter=Counters_symb[i])
            leafs.append(node)
            q.put(node)
    while (q.qsize() >= 2):
        left_child = q.get()
        right_child = q.get()
        parent = Node(counter = left_child.counter + right_child.counter)
        parent.left_child = left_child
        parent.right_child = right_child
        left_child.parent = parent
        right_child.parent = parent
        q.put(parent)
    codes = {}
    for leaf in leafs:
        node = leaf
        code = ""
        while node.parent != None:
            if node.parent.left_child == node:
                code = "0" + code
            else:
                code = "1" + code
            node = node.parent
        codes[leaf.symbol] = code
    coded_message = ""
    for s in text:
        coded_message += codes[s]
    ##k = 8 - len(coded_message) % 8
    coded_message += (8 - len(coded_message) % 8) * "0"
    bytes_string = b""
    for i in range(0, len(coded_message), 8):
        s = coded_message[i:i + 8]
        x = string_binary_to_int(s)
        bytes_string += x.to_bytes(1, "big")
    return bytes_string, codes, n
def HA_decoding(compressed_text, codes, n):
    reverse_codes = {x: y for y,x in codes.items()}
    res = bytearray(b'')
    text = bytes_to_binary(compressed_text)
    current = ''
    i = 0
    while (len(res)<n and i < len(text)):
        current += text[i]
        if (current in reverse_codes.keys()):
            res.append(reverse_codes[current])
            current = ''
        i +=1
    return res
def HA_decode(compressed_text, codes, n):
    l = compressed_text[0]

    bit_data = compressed_text[1:]
    bit_string = ''.join(format(byte, '08b') for byte in bit_data)
    if l:
        bit_string = bit_string[:-l]
    reverse_codes = {x: y for y, x in codes.items()}
    current = ''
    res = bytearray(b'')
    for i in bit_string:
        current+= i
        if (current in reverse_codes):
            res.append(reverse_codes[current])
            current = ''
    return res
def string_binary_to_int(s):
    X = 0
    for i in range(8):
        if s[i] == "1":
            X = X + 2**(7-i)
    return X
def bytes_to_binary(n):
    # Преобразуем байты в двоичную строку
    return ''.join(format(byte, '08b') for byte in n)

def LZ77(text, buffer_size):
    string_size = buffer_size
    coding_list = []
    buffer = bytearray(b"")
    n = len(text)
    i = 0
    while i < n:
        buffer = text[max(0,i-buffer_size) : i]
        #print(i, repr(buffer))
        new_buffer_size = len(buffer)
        shift = -1
        for j in range(string_size, -1, -1):
            subS = text[i : min(i + j,n)]
            shift = buffer.find(subS)
            if shift != -1:
                break
                # Если найдено совпадение
        if shift != -1:
            next_symbol_index = i + len(subS)
            if next_symbol_index < n:
                next_symbol = text[next_symbol_index]
            else:
                next_symbol = 1
            #print(new_buffer_size - shift, len(subS), next_symbol)
            coding_list.append((new_buffer_size - shift, len(subS), next_symbol))
            i += len(subS) + 1
        else:
            coding_list.append((0, 0, text[i]))
            i += 1
    res = bytearray(b'')
    for i in coding_list:
        d = (str(i).replace('(','').replace(')','').replace(' ',''))
        #print(d)
        res.extend(d.encode('utf-8'))
        res.extend(b' ')
    return (res)
def LZ77_decoding(compressed_text):
    text = bytearray(b"")
    compressed = []
    compressed = compressed_text.split(b' ')
    for i in compressed[:-1]:
        offset, length, symbol = (i.split(b','))
        offset = int(offset.decode('utf-8'))  # Декодируем из байтов в строку и затем конвертируем в int
        length = int(length.decode('utf-8'))  # Декодируем из байтов в строку и затем конвертируем в int
        symbol = (int((symbol).decode('utf-8')))
        if length == 0:
            text.append(symbol)
            continue
        start = len(text) - offset
        if start < 0:
            raise ValueError("Invalid offset in compressed data")
        for i in range(length):
            text.append(text[start + i])
        text.append(symbol)
    return text[:-1]

def LZ78(text):
    dictionary = {b'': 0}
    current = b''
    encoded = []
    code = 1
    for i in text:
        current_new = current + bytes([i])
        if current_new in dictionary:
            current = current_new
        else:
            encoded.append((dictionary[current], bytes([i])))
            dictionary[current_new] = code
            code += 1
            current = b''

    res = bytearray(b'')
    for code_num, char in encoded:
        res += struct.pack('>I', code_num)
        res += char

    return bytes(res)

def LZ78_decoding(compressed_data):
    dictionary = {0: b''}
    decoded = bytearray()
    code = 1

    # Чтение данных по 5 байт (4 байта код + 1 байт символ)
    for i in range(0, len(compressed_data), 5):
        chunk = compressed_data[i:i + 5]
        if len(chunk) < 5:
            break

        # Распаковка кода и символа
        current_code = struct.unpack('>I', chunk[:4])[0]
        current_char = chunk[4:5]

        # Получение фразы из словаря
        if current_code not in dictionary:
            raise ValueError(f"Invalid code {current_code}")

        phrase = dictionary[current_code] + current_char
        decoded += phrase

        # Добавление новой фразы в словарь
        dictionary[code] = phrase
        code += 1

    return bytes(decoded)

def png_to_raw(image_path, output_path):
    image = Image.open(image_path)
    if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
        # Удаляем альфа-канал
        image = image.convert('RGB')

    raw_pixels = numpy.array(image)
    raw_data = raw_pixels.tobytes()

    with open(output_path, 'wb') as f:
        f.write(raw_data)

'''png_to_raw('color.png', 'color.raw')
png_to_raw('gray.png', 'gray.raw')
png_to_raw('чб.png', 'black_white.raw')'''
#file_input = open("test.txt","rb")
#file_output = open("compressed_test.txt","wb")
#text = (file_input.read()).replace(b'\r', b'').replace(b'\t', b'').replace(b'\n', b'')
#text = b'banana'
#text = text.encode('utf-8')
#text1 = b'abacababacabc'
#compressed_text = rle_encoding(text)
#decompressed_text = rle_decoding(compressed_text)
#file_output.write(compressed_text)
#last_column, text_index = BWT(text)
#print(text_index, last_column)
#print(better_BWT_decoding(last_column, text_index))
#comp_text, codes, n =Haffman_alg(text)
#print(HA_decoding(comp_text, codes, n))
#print(Haffman_alg(text))
#print(compressed_text)
#print(decompressed_text)
#print(MTF(text))
#print(MTF_decoding(MTF(text)))
#file_input.close()
#print(LZ78_decoding(LZ78(text1)))
#### КОМПРЕССОРЫ
'''name_file = 'black_white'
name_comp = 'LZ77+HA'

file_in1 = open('compressed_'+name_file+'_LZ77.txt', 'rb')
file_out1 = open('compressed_'+name_file+'_'+name_comp+'.txt','wb')

text1 = file_in1.read()
start_time = time.time()'''
#----------------------------------------------------------------------------------------
##RLE--------------------------------------------------------------------------------------

'''compressed = new_rle(text1)
file_out1.write(compressed)'''
#HA-----------------------------------------------------------------------------------------------
##HAffman Algoritm
'''compressed, codes1, n1 = Haffman_alg(text1)

file_out1.write(bytes(str(n1) + '\n', 'utf-8'))
file_out1.write(bytes(str(len(codes1)) + '\n', 'utf-8'))
for i in codes1.keys():
    file_out1.write(bytes(f"{i}:{codes1[i]}\n", 'utf-8'))
file_out1.write(b'\n')
file_out1.write(compressed)'''
##блоки хаффман
'''block_size_ha = 10000
file_out1.write(str(len(text1)//block_size_ha+1).encode('utf-8'))
file_out1.write(b'\n')
file_out1.write(str(block_size_ha).encode('utf-8'))
file_out1.write(b'\n')
for i in range(len(text1)//block_size_ha+1):
    text_part = text1[i*block_size_ha:min(i*block_size_ha+block_size_ha, len(text1))]
    #print(text_part)
    if (text_part!=b''):
        compressed, codes1, n1 = Haffman_alg(text_part)
        file_out1.write(bytes(str(n1) + '\n', 'utf-8'))
        file_out1.write(bytes(str(len(codes1)) + '\n', 'utf-8'))
        for i in codes1.keys():
            file_out1.write(bytes(f"{i}:{codes1[i]}\n", 'utf-8'))
    file_out1.write(str(len(compressed)).encode('utf-8'))
    file_out1.write(b'\n')
    file_out1.write(compressed)
    file_out1.write(b'\n')'''
#BWT+RLE--------------------------------------------------------------------------------------
#BWT
'''block_size = 10000
file_out1.write(str(block_size).encode('utf-8'))
file_out1.write(b'\n')
after_bwt = bytearray(b'')
indexes = []
for i in range (0,len(text1)//block_size+1):
    text_part = text1[i*block_size:min(i*block_size+block_size, len(text1))]
    if(text_part!= b''):
        bwt_comp, text_index = BWT(text_part)
        after_bwt.extend(bwt_comp)
        indexes.append(text_index)

compressed= new_rle(after_bwt)
#print(block_size, indexes)
indexes_byte = [str(i).encode('utf-8') for i in indexes ]
for i in range(len(indexes_byte)):
    file_out1.write(indexes_byte[i])
    file_out1.write(b' ')
file_out1.write(b'\n')

file_out1.write(compressed)'''
#BWT+MTF+HA---------------------------------------------------------------------------------------------
'''for k in range (17000,22000,2000):
    block_size = k
    file_out1.write(str(block_size).encode('utf-8'))
    file_out1.write(b'\n')
    after_bwt = bytearray(b'')
    indexes = []
    for i in range (0,len(text1)//block_size+1):
        text_part = text1[i*block_size:min(i*block_size+block_size, len(text1))]
        if(text_part!= b''):
            bwt_comp, text_index = BWT(text_part)
            after_bwt.extend(bwt_comp)
            indexes.append(text_index)

    text_MTF = MTF(after_bwt)
    print(block_size, entropy(text_MTF))
#print(block_size, indexes)'''

'''indexes_byte = [str(i).encode('utf-8') for i in indexes ]
for i in range(len(indexes_byte)):
    file_out1.write(indexes_byte[i])
    file_out1.write(b' ')
file_out1.write(b'\n')

compressed, codes1, n1 = Haffman_alg(text_MTF)

file_out1.write(bytes(str(n1) + '\n', 'utf-8'))
file_out1.write(bytes(str(len(codes1)) + '\n', 'utf-8'))
for i in codes1.keys():
    file_out1.write(bytes(f"{i}:{codes1[i]}\n", 'utf-8'))

file_out1.write(compressed)'''
##блоки хаффман
'''block_size_ha = 10000
file_out1.write(str(len(text_MTF)//block_size_ha+1).encode('utf-8'))
file_out1.write(b'\n')
file_out1.write(str(block_size_ha).encode('utf-8'))
file_out1.write(b'\n')
for i in range(len(text_MTF)//block_size_ha+1):
    text_part = text_MTF[i*block_size_ha:min(i*block_size_ha+block_size_ha, len(text_MTF))]
    #print(text_part)
    if (text_part!=b''):
        compressed, codes1, n1 = Haffman_alg(text_part)
        file_out1.write(bytes(str(n1) + '\n', 'utf-8'))
        file_out1.write(bytes(str(len(codes1)) + '\n', 'utf-8'))
        for i in codes1.keys():
            file_out1.write(bytes(f"{i}:{codes1[i]}\n", 'utf-8'))
    file_out1.write(str(len(compressed)).encode('utf-8'))
    file_out1.write(b'\n')
    file_out1.write(compressed)
    file_out1.write(b'\n')
'''
##BWT+MTF+RLE+HA--------------------------------------------------------------------------------
'''block_size = 10000
file_out1.write(str(block_size).encode('utf-8'))
file_out1.write(b'\n')
after_bwt = bytearray(b'')
indexes = []
for i in range (0,len(text1)//block_size+1):
    text_part = text1[i*block_size:min(i*block_size+block_size, len(text1))]
    if(text_part!= b''):
        bwt_comp, text_index = BWT(text_part)
        after_bwt.extend(bwt_comp)
        indexes.append(text_index)

text_MTF = MTF(after_bwt)
text_RLE = new_rle(text_MTF)
#print(block_size, indexes)
indexes_byte = [str(i).encode('utf-8') for i in indexes ]
for i in range(len(indexes_byte)):
    file_out1.write(indexes_byte[i])
    file_out1.write(b' ')
file_out1.write(b'\n')'''

'''compressed, codes1, n1 = Haffman_alg(text_RLE)

file_out1.write(bytes(str(n1) + '\n', 'utf-8'))
file_out1.write(bytes(str(len(codes1)) + '\n', 'utf-8'))
for i in codes1.keys():
    file_out1.write(bytes(f"{i}:{codes1[i]}\n", 'utf-8'))

file_out1.write(compressed)'''
##блоки хаффман
'''block_size_ha = 10000
file_out1.write(str(len(text_RLE)//block_size_ha+1).encode('utf-8'))
file_out1.write(b'\n')
file_out1.write(str(block_size_ha).encode('utf-8'))
file_out1.write(b'\n')
for i in range(len(text_RLE)//block_size_ha+1):
    text_part = text_RLE[i*block_size_ha:min(i*block_size_ha+block_size_ha, len(text_RLE))]
    #print(text_part)
    if (text_part!=b''):
        compressed, codes1, n1 = Haffman_alg(text_part)
        file_out1.write(bytes(str(n1) + '\n', 'utf-8'))
        file_out1.write(bytes(str(len(codes1)) + '\n', 'utf-8'))
        for i in codes1.keys():
            file_out1.write(bytes(f"{i}:{codes1[i]}\n", 'utf-8'))
    file_out1.write(str(len(compressed)).encode('utf-8'))
    file_out1.write(b'\n')
    file_out1.write(compressed)
    file_out1.write(b'\n')'''

##LZ77-------------------------------------------------------------------------------------------
'''compressed_text = LZ77(text1, 4096)
file_out1.write(compressed_text)'''

#print(compressed_text)
##LZ77+HA-------------------------------------------------------------------------------------------
'''text_LZ77 = LZ77(text1)
compressed, codes1, n1 = Haffman_alg(text_LZ77)

file_out1.write(bytes(str(n1) + '\n', 'utf-8'))
file_out1.write(bytes(str(len(codes1)) + '\n', 'utf-8'))
for i in codes1.keys():
    file_out1.write(bytes(f"{i}:{codes1[i]}\n", 'utf-8'))
file_out1.write(b'\n')
file_out1.write(compressed)'''
##блоки хаффман
'''block_size_ha = 10000
file_out1.write(str(len(text_LZ77)//block_size_ha+1).encode('utf-8'))
file_out1.write(b'\n')
file_out1.write(str(block_size_ha).encode('utf-8'))
file_out1.write(b'\n')
for i in range(len(text_LZ77)//block_size_ha+1):
    text_part = ttext_LZ77[i*block_size_ha:min(i*block_size_ha+block_size_ha, len(text_LZ77))]
    #print(text_part)
    if (text_part!=b''):
        compressed, codes1, n1 = Haffman_alg(text_part)
        file_out1.write(bytes(str(n1) + '\n', 'utf-8'))
        file_out1.write(bytes(str(len(codes1)) + '\n', 'utf-8'))
        for i in codes1.keys():
            file_out1.write(bytes(f"{i}:{codes1[i]}\n", 'utf-8'))
    file_out1.write(str(len(compressed)).encode('utf-8'))
    file_out1.write(b'\n')
    file_out1.write(compressed)
    file_out1.write(b'\n')'''


##LZ78------------------------------------------------------------------------------------------
'''compressed_text = LZ78(text1)
file_out1.write(compressed_text)'''
##LZ78+HA---------------------------------------------------------------------------------------
'''text_LZ78 = LZ78(text1)
compressed, codes1, n1 = Haffman_alg(text_LZ78)

file_out1.write(bytes(str(n1) + '\n', 'utf-8'))
file_out1.write(bytes(str(len(codes1)) + '\n', 'utf-8'))
for i in codes1.keys():
    file_out1.write(bytes(f"{i}:{codes1[i]}\n", 'utf-8'))
file_out1.write(compressed)'''
##блоки хаффман
'''block_size_ha = 10000
file_out1.write(str(len(text_LZ78)//block_size_ha+1).encode('utf-8'))
file_out1.write(b'\n')
file_out1.write(str(block_size_ha).encode('utf-8'))
file_out1.write(b'\n')
for i in range(len(text_LZ78)//block_size_ha+1):
    text_part = text_LZ78[i*block_size_ha:min(i*block_size_ha+block_size_ha, len(text_LZ78))]
    #print(text_part)
    if (text_part!=b''):
        compressed, codes1, n1 = Haffman_alg(text_part)
        file_out1.write(bytes(str(n1) + '\n', 'utf-8'))
        file_out1.write(bytes(str(len(codes1)) + '\n', 'utf-8'))
        for i in codes1.keys():
            file_out1.write(bytes(f"{i}:{codes1[i]}\n", 'utf-8'))
    file_out1.write(str(len(compressed)).encode('utf-8'))
    file_out1.write(b'\n')
    file_out1.write(compressed)
    file_out1.write(b'\n')'''
#--------------------------------------------------------------------------------------------

'''end_time = time.time()
print(end_time - start_time)
print()
file_in1.close()
file_out1.close()'''

#**********************************************************************************************************

# декомпрессия
'''name_file = 'black_white'
name_comp = 'LZ77+HA'
file_in1 = open('compressed_'+name_file+'_'+name_comp+'.txt', 'rb')
file_out1 = open('decompressed_'+name_file+'_'+name_comp+'.raw','wb')

start_time = time.time()'''
#---------------------------------------------------------------------------------------------------
##RLE-----------------------------------------------------------------------------------------
'''text_compressed = file_in1.read()
decompressed_ = new_rle_decoding(text_compressed)
file_out1.write(decompressed_)'''

##HA------------------------------------------------------------------------------------------

##Haffman_algoritm
'''n = int(file_in1.readline())
codes_count  = file_in1.readline()
codes = {}
for i in range (int(codes_count)):
    value, code = str(file_in1.readline().replace(b'\n', b'')).replace("b'", '').replace("'", '').split(':')
    print(value, code)
    codes[int(value)] = code

text_compressed = (file_in1.read())
decompressed_HA = HA_decoding(text_compressed, codes, n)
file_out1.write(decompressed_HA)'''


##Haffman блоки
'''N = int(file_in1.readline())
block_size = int(file_in1.readline())
decompressed_HA = bytearray(b'')
for i in range(N-1):
     n = file_in1.readline()
     if (n != b'\n'):
        n = int(n)
     else:
        print(b'\n', i)
        n = file_in1.readline()
        n = int(n)
     print(n, i)
     codes_count = file_in1.readline()
     codes = {}
     for i in range(int(codes_count)):
         value, code = str(file_in1.readline().replace(b'\n', b'')).replace("b'", '').replace("'", '').split(':')
         # print(value, code)
         codes[int(value)] = code

     part_text = bytearray(b'')

     len_text_part = file_in1.readline()
     while (len(part_text) < int(len_text_part)):
         part_text.extend(file_in1.readline())
     decompressed_HA.extend(HA_decoding(part_text, codes, n))

file_out1.write(decompressed_HA)

'''
##BWT+RLE----------------------------------------------------------------------

'''block_size = int(str(file_in1.readline()).replace("b'", '').replace("'", '').replace('\\n',''))
text_indexes = []
indexes = str(file_in1.readline()).replace("b'", '').replace("'", '').replace('\\n','').split(' ')
for s in indexes:
    if(s!= ''): text_indexes.append(int(s))

text_compressed = file_in1.read()
decompressed_ = new_rle_decoding(text_compressed)
#BWT

#indexes = file_in1.readline()
text_decompressed = bytearray(b'')
n = len(decompressed_)
for i in range(len(text_indexes)):
    text_part = decompressed_[i*block_size:min(i*block_size+block_size, n)]
    text_decompressed.extend(better_BWT_decoding(text_part, text_indexes[i]))
    print(text_indexes)
file_out1.write(text_decompressed)'''

##BWT+MTF+HA---------------------------------------------------------------------------------
##Haffman_algoritm
'''block_size_BWT = int(str(file_in1.readline()).replace("b'", '').replace("'", '').replace('\\n',''))
text_indexes = []
indexes = str(file_in1.readline()).replace("b'", '').replace("'", '').replace('\\n','').split(' ')
for s in indexes:
    if(s!= ''): text_indexes.append(int(s))'''
#---
'''n = int(file_in1.readline())
codes_count  = file_in1.readline()
codes = {}
for i in range (int(codes_count)):
    value, code = str(file_in1.readline().replace(b'\n', b'')).replace("b'", '').replace("'", '').split(':')
    #print(value, code)
    codes[int(value)] = code

text_compressed = (file_in1.read())
decompressed_HA = HA_decoding(text_compressed, codes, n)
file_out.write(decompressed_HA)'''
#print(decompressed_HA)
#file_out1.write(decompressed_HA)'''

##Haffman блоки
'''N = int(file_in1.readline())
block_size = int(file_in1.readline())
decompressed_HA = bytearray(b'')
for i in range(N):
     n = file_in1.readline()
     if (n != b'\n'):
        n = int(n)
     else:
        print(b'\n', i)
        n = file_in1.readline()
        n = int(n)
     print(n, i)
     codes_count = file_in1.readline()
     codes = {}
     for i in range(int(codes_count)):
         value, code = str(file_in1.readline().replace(b'\n', b'')).replace("b'", '').replace("'", '').split(':')
         # print(value, code)
         codes[int(value)] = code

     part_text = bytearray(b'')

     len_text_part = file_in1.readline()
     while (len(part_text) < int(len_text_part)):
         part_text.extend(file_in1.readline())
     decompressed_HA.extend(HA_decoding(part_text, codes, n))


'''

'''decompressed_ = MTF_decoding(decompressed_HA)
#file_out.write(decompressed_)
#BWT

text_decompressed = bytearray(b'')
n = len(decompressed_)
for i in range(len(text_indexes)):
    text_part = decompressed_[i*block_size_BWT:min(i*block_size_BWT+block_size_BWT, n)]
    text_decompressed.extend(better_BWT_decoding(text_part, text_indexes[i]))
    #print(text_indexes)
file_out1.write(text_decompressed)'''
##BWT+MTF+RLE+HA--------------------------------------------------------------------------------
##Haffman_algoritm
'''block_size_BWT = int(str(file_in1.readline()).replace("b'", '').replace("'", '').replace('\\n',''))
text_indexes = []

indexes = str(file_in1.readline()).replace("b'", '').replace("'", '').replace('\\n','').split(' ')
for s in indexes:
    if(s!= ''): text_indexes.append(int(s))'''

#---
'''n = int(file_in1.readline())
codes_count  = file_in1.readline()
codes = {}
for i in range (int(codes_count)):
    value, code = str(file_in1.readline().replace(b'\n', b'')).replace("b'", '').replace("'", '').split(':')
    #print(value, code)
    codes[int(value)] = code

text_compressed = (file_in1.read())
decompressed_HA = HA_decoding(text_compressed, codes, n)'''
#file_out.write(decompressed_HA)
#print(decompressed_HA)
#file_out1.write(decompressed_HA)

##Haffman блоки
'''N = int(file_in1.readline())
block_size = int(file_in1.readline())
decompressed_HA = bytearray(b'')
for i in range(N):
    n = file_in1.readline()
    if (n != b'\n'):
        n = int(n)
    else:
        print(b'\n', i)
        n = file_in1.readline()
        n = int(n)
    print(n, i)
    codes_count = file_in1.readline()
    codes = {}
    for i in range(int(codes_count)):
        value, code = str(file_in1.readline().replace(b'\n', b'')).replace("b'", '').replace("'", '').split(':')
        # print(value, code)
        codes[int(value)] = code

    part_text = bytearray(b'')

    len_text_part = file_in1.readline()
    while (len(part_text) < int(len_text_part)):
        part_text.extend(file_in1.readline())
    decompressed_HA.extend(HA_decoding(part_text, codes, n))


decompressed_RLE = new_rle_decoding(decompressed_HA)
decompressed_ = MTF_decoding(decompressed_RLE)
#file_out.write(decompressed_)
#BWT

text_decompressed = bytearray(b'')
n = len(decompressed_)
for i in range(len(text_indexes)):
    text_part = decompressed_[i*block_size_BWT:min(i*block_size_BWT+block_size_BWT, n)]
    text_decompressed.extend(better_BWT_decoding(text_part, text_indexes[i]))
file_out1.write(text_decompressed)'''
##LZ77-------------------------------------------------------------------------------------------

'''n = int(file_in1.readline())
codes_count  = file_in1.readline()
codes = {}
for i in range (int(codes_count)):
    value, code = str(file_in1.readline().replace(b'\n', b'')).replace("b'", '').replace("'", '').split(':')
    #print(value, code)
    codes[int(value)] = code

text_compressed = (file_in1.read())
decompressed_HA = HA_decoding(text_compressed, codes, n)'''
#file_out.write(decompressed_HA)
#print(decompressed_HA)
#file_out1.write(decompressed_HA)

##Haffman блоки
'''N = int(file_in1.readline())
block_size = int(file_in1.readline())
decompressed_HA = bytearray(b'')
for i in range(N):
    n = file_in1.readline()
    if (n != b'\n'):
        n = int(n)
    else:
        print(b'\n', i)
        n = file_in1.readline()
        n = int(n)
    #print(n, i)
    codes_count = file_in1.readline()
    codes = {}
    for i in range(int(codes_count)):
        value, code = str(file_in1.readline().replace(b'\n', b'')).replace("b'", '').replace("'", '').split(':')
        # print(value, code)
        codes[int(value)] = code

    part_text = bytearray(b'')

    len_text_part = file_in1.readline()
    while (len(part_text) < int(len_text_part)):
        part_text.extend(file_in1.readline())
    decompressed_HA.extend(HA_decoding(part_text, codes, n))
'''
'''decompressed = LZ78_decoding(decompressed_HA)
file_out1.write(decompressed)'''
##LZ77+HA-------------------------------------------------------------------------------------------
#Haffman блоки
'''N = int(file_in1.readline())
block_size = int(file_in1.readline())
decompressed_HA = bytearray(b'')
for i in range(N):
    n = file_in1.readline()
    if (n != b'\n'):
        n = int(n)
    else:
        print(b'\n', i)
        n = file_in1.readline()
        n = int(n)
    #print(n, i)
    codes_count = file_in1.readline()
    codes = {}
    for i in range(int(codes_count)):
        value, code = str(file_in1.readline().replace(b'\n', b'')).replace("b'", '').replace("'", '').split(':')
        # print(value, code)
        codes[int(value)] = code

    part_text = bytearray(b'')

    len_text_part = file_in1.readline()
    while (len(part_text) < int(len_text_part)):
        part_text.extend(file_in1.readline())
    decompressed_HA.extend(HA_decoding(part_text, codes, n))

decompressed = LZ77_decoding(decompressed_HA)
file_out1.write(decompressed)'''
##LZ78------------------------------------------------------------------------------------------
##LZ78+HA---------------------------------------------------------------------------------------


#---------------------------------------------------------------------------------------------------
'''end_time = time.time()
print(end_time - start_time)
file_in1.close()
file_out1.close()'''

name_file = 'black_white'
format ='.raw'
name_comp = 'LZ77+HA'
file_in = open(name_file + format, 'rb')
file_in1 = open('compressed_'+name_file+ '_' +name_comp+'.txt', 'rb')
file_in2 = open('decompressed_'+name_file+'_'+name_comp + format, 'rb')
text = file_in.read()
text1 = file_in1.read()
text2 = file_in2.read()
print(len(text), len(text1), len(text2), len(text)/len(text1))