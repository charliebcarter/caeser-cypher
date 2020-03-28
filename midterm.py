from pyspark import SparkContext, SparkConf
import nltk

nltk.download('omw')
from nltk.corpus import wordnet as wn
import sys

conf = SparkConf().setAppName("Midterm").setMaster("local")
sc = SparkContext(conf=conf)


#check if the word exists in wordnet
def checkword(word):
    if (wn.synsets(word)):
        return True
    else:
        return False

# master function to iterate through different shifts and find the correct shift with checkword function
def decryptIterative(word):
    for x in range(25):
        shifted = ceaserShift(word, (x + 25) % 25)
        if (checkword(shifted) == True):
            return x
    return "error max shift"

# shifts text by "s" ignoring ASCII that are not encrypted or not relevant. Returns the shifted text
def ceaserShift(text, s):
    result = ""
    # transverse the plain text
    for i in range(len(text)):
        char = text[i]
        # skip decryption for whitespace
        if ((ord(char) < 65) or (ord(char) > 90 and ord(char) < 97) or (ord(char) > 122)):
            result += char
        # Encrypt uppercase characters in plain text
        else:
            if (char.isupper()):
                result += chr((ord(char) - s - 65) % 26 + 65)
            # Encrypt lowercase characters in plain text
            else:
                result += chr((ord(char) - s - 97) % 26 + 97)
# returning shifted word
    return result

# function to find distance (shift) between characters and what they are expected to be based on NLTK data. Returns an average of the distances for each character that represents an estimated shift.
def avg_distance(input):
    avg = 0
    char_list = [ord(lis[0]) for lis in input]
    actualFreq = [float(lis[1][0]) for lis in input]
    expFreq = [float(lis[1][1]) for lis in input]

    for i in range(len(input)):
        temp = actualFreq[i]
        absolute_difference_function = lambda list_value: abs(list_value - temp)
        closest = min(expFreq, key=absolute_difference_function)
        index_of_closest_value = expFreq.index(closest)
        distance = char_list[index_of_closest_value]-char_list[i]

        if (distance<0):
            distance = (abs(distance)+26)

        avg += abs(distance)
       # print(chr(char_list[index_of_closest_value]))
       # print(closest)
       # print(distance)
    return(round(avg/26))


#testing the above functions
# print(checkword("charles"))
# print(decryptIterative("mbfxl"))
# print(ceaserShift("johyslz pz mybzayhalk dpao wfaovu! ~_~ ^:?_?:^", 7))

# load encrypted text file to RDD
fileToAnalyze = sc.textFile('Encrypted-1.txt')

fileToAnalyze1 = sc.textFile('Encrypted-2.txt')

fileToAnalyze2 = sc.textFile('Encrypted-3.txt')

expCharFreqs = sc.parallelize([(u'a', u'8.167'),(u'b', u'1.492'),(u'c', u'2.202'),(u'd', u'4.253'),(u'e', u'12.702'),(u'f', u'2.228'),(u'g', u'2.015'),(u'h', u'6.094'),('i', u'6.966'),(u'j', u'0.153'),(u'k', u'1.292'),(u'l', u'4.025'),(u'm', u'2.406'),(u'n', u'6.749'),(u'o', u'7.507'),(u'p', u'1.929'),(u'q', u'0.095'),(u'r', u'5.987'),(u's', u'6.327'),(u't', u'9.356'),(u'u', u'2.758'),(u'v', u'0.978'),(u'w', u'2.560'),(u'x', u'0.150'),(u'y', u'1.994'),(u'z', u'0.077')])

# creating rdds to compare the char frequencies with those that are to be expected in the english language
charUnfiltered = fileToAnalyze.flatMap(lambda line: list(line.lower()))
charFiltered = charUnfiltered.filter(lambda char: not ((ord(char) < 65) or (ord(char) > 90 and ord(char) < 97) or (ord(char) > 122)))
totalChar = charFiltered.count()

charFreqs = charFiltered.map(lambda word: (word, 1)).reduceByKey(lambda a,b:a +b)
charFreqsPercentage = charFreqs.map(lambda char: (char[0], '{:.3}'.format((char[1]/totalChar)*100)))
comparison = charFreqsPercentage.join(expCharFreqs)
test_list = (comparison.collect())


charUnfiltered1 = fileToAnalyze1.flatMap(lambda line: list(line.lower()))
charFiltered1 = charUnfiltered1.filter(lambda char: not ((ord(char) < 65) or (ord(char) > 90 and ord(char) < 97) or (ord(char) > 122)))
totalChar1 = charFiltered1.count()

charFreqs1 = charFiltered1.map(lambda word: (word, 1)).reduceByKey(lambda a,b:a +b)
charFreqsPercentage1 = charFreqs1.map(lambda char: (char[0], '{:.3}'.format((char[1]/totalChar1)*100)))
comparison1 = charFreqsPercentage1.join(expCharFreqs)
test_list1 = (comparison1.collect())


charUnfiltered2 = fileToAnalyze2.flatMap(lambda line: list(line.lower()))
charFiltered2 = charUnfiltered2.filter(lambda char: not ((ord(char) < 65) or (ord(char) > 90 and ord(char) < 97) or (ord(char) > 122)))
totalChar2 = charFiltered2.count()

charFreqs2 = charFiltered2.map(lambda word: (word, 1)).reduceByKey(lambda a,b:a +b)
charFreqsPercentage2 = charFreqs2.map(lambda char: (char[0], '{:.3}'.format((char[1]/totalChar2)*100)))
comparison2 = charFreqsPercentage2.join(expCharFreqs)
test_list2 = (comparison2.collect())

# split file into words. Lambda function is converted all words to lowercase and splitting by space

wordSplit = fileToAnalyze.map(lambda word: str(word).lower()).flatMap(lambda word: word.split(" "))

wordSplit1 = fileToAnalyze1.map(lambda word: str(word).lower()).flatMap(lambda word: word.split(" "))

wordSplit2 = fileToAnalyze2.map(lambda word: str(word).lower()).flatMap(lambda word: word.split(" "))
# filter empty lines. Lambda function is removing elements of size 0
non_empty_lines = wordSplit.filter(lambda x: len(x) > 0)

non_empty_lines1 = wordSplit1.filter(lambda x: len(x) > 0)

non_empty_lines2 = wordSplit2.filter(lambda x: len(x) > 0)

# take first element of filtered RDD
sampleRDD = non_empty_lines.first()
sampleRDD1 = non_empty_lines1.first()
sampleRDD2 = non_empty_lines2.first()
# find correct shift from sample word taken from filter RDD
correctShift = decryptIterative(sampleRDD)
correctShift1 = decryptIterative(sampleRDD1)
correctShift2 = decryptIterative(sampleRDD2)
# test to make sure correct shift was found
print(correctShift)
# performing ceasershift on RDD with the shift value taken from the sample RDD
unalteredShift = fileToAnalyze.map(lambda word: ceaserShift(word, correctShift))
unalteredShift1 = fileToAnalyze1.map(lambda word: ceaserShift(word, correctShift1))
unalteredShift2 = fileToAnalyze2.map(lambda word: ceaserShift(word, correctShift2))

estShift = fileToAnalyze.map(lambda word: ceaserShift(word, avg_distance(test_list)))
estShift1 = fileToAnalyze1.map(lambda word: ceaserShift(word, avg_distance(test_list1)))
estShift2 = fileToAnalyze2.map(lambda word: ceaserShift(word, avg_distance(test_list2)))

print("Number of words in file 1: " + str(non_empty_lines.count()))
print("Number of characters in Encrypted-1.txt: "+ str(totalChar))
print("Estimated shift for Encrypted-1.txt: "+str(avg_distance(test_list)))
print("Actual shift of Encrypted-1.txt: "+ str(correctShift))
print()
print("Number of words in file 2: " + str(non_empty_lines1.count()))
print("Number of characters in Encrypted-2.txt: "+ str(totalChar1))
print("Estimated shift for Encrypted-2.txt: "+str(avg_distance(test_list1)))
print("Actual shift of Encrypted-2.txt: "+ str(correctShift1))
print()
print("Number of words in file 3: " + str(non_empty_lines2.count()))
print("Number of characters in Encrypted-3.txt: "+ str(totalChar2))
print("Estimated shift for Encrypted-3.txt: "+str(avg_distance(test_list2)))
print("Actual shift of Encrypted-3.txt: "+ str(correctShift2))
print()
# test RDD of words that are split into their own lines and made lowercase. This could stand some tweaking to save proccessing
# decryptedRDD = non_empty_lines.map(lambda word: ceaserShift(word, correctShift))
# more test printing
# print(ceaserShift())
# saving sample RDD to text for debugging purposes
#non_empty_lines.saveAsTextFile("/home/charles/Downloads/spark/MIDTERMOUTPUT")
# saving the unmolested encryption files with only the shift applied. Formatting and capitilisation is unchanged from original text file

unalteredShift.saveAsTextFile("/home/charles/Downloads/spark/CORRECT_DECRYPTION/Encrypted-1")
unalteredShift1.saveAsTextFile("/home/charles/Downloads/spark/CORRECT_DECRYPTION/Encrypted-2")
unalteredShift2.saveAsTextFile("/home/charles/Downloads/spark/CORRECT_DECRYPTION/Encrypted-3")

estShift.saveAsTextFile("/home/charles/Downloads/spark/ESTIMATED_DECRYPTION/Encrypted-1")
estShift1.saveAsTextFile("/home/charles/Downloads/spark/ESTIMATED_DECRYPTION/Encrypted-2")
estShift2.saveAsTextFile("/home/charles/Downloads/spark/ESTIMATED_DECRYPTION/Encrypted-3")
# debugging info while developing avg_distance function
# expCharFreqs.saveAsTextFile("/home/charles/Downloads/spark/DECRYPTION/Expected_Character_Frequencies")
# charFreqsPercentage.saveAsTextFile("/home/charles/Downloads/spark/DECRYPTION/Character_Frequencies")
# comparison.coalesce(1).saveAsTextFile("/home/charles/Downloads/spark/DECRYPTION/comparison")
