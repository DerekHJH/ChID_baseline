userPath = "/home/heyibo/nlp/project/"
trainDataPath = userPath + "ChID/train_data.json"
testDataPath = userPath + "ChID/test_data.json"

#return the length of the longest common subsequence
def LCS(str1, str2):
    len1 = len(str1)
    len2 = len(str2)
    res = [ [0 for i in range(len1+1)] for j in range(len2+1)]
    for i in range(1, len2+1):
        for j in range(1, len1+1):
            if(str2[i-1] == str1[j-1]):
                res[i][j] = 1 + res[i-1][j-1]
            else:
                res[i][j] = max(res[i-1][j], res[i][j-1])
    return res[-1][-1]

#return edit distance
def EditDistance(str1, str2):
    len1 = len(str1)
    len2 = len(str2)
    res = [ [i+j for j in range(len2+1)] for i in range(len1+1)]
    for i in range(1, len1+1):
        for j in range(1, len2+1):
            if(str1[i-1] == str2[j-1]):
                d = 0
            else:
                d = 1
            res[i][j] = min(res[i-1][j]+1, res[i][j-1]+1, res[i-1][j-1]+d)
    return res[len1][len2]

#calculate name similarity
def Name_Similarity(testMethod, method):
    len_nt = len(testMethod)
    len_nf = len(method)
    lcs = LCS(testMethod, method)
    LCS_B = lcs*1.0/(max(len_nf, len_nt))
    LCS_U = lcs*1.0/len_nf
    Score_Edit = 1 - EditDistance(testMethod, method)*1.0/(max(len_nf, len_nt))
    return LCS_B, LCS_U, Score_Edit

if __name__ == "__main__":
    trainData = open(trainDataPath, "r", encoding="utf-8")
    testData = open(testDataPath, "r", encoding="utf-8")

    #content
    trainLines = trainData.readlines()
    contentList_train = []
    for line in trainLines:
        line = eval(line)
        contentList_train.append(line["content"])

    testLines = testData.readlines()
    LCS_B_scores = []
    LCS_U_scores = []
    Edit_scores = []
    k = 0
    total_test = len(testLines)
    lcsu = 0
    lcsb = 0
    editscore = 0
    threshold = 0.8

    for line in testLines:
        line = eval(line)
        content = line["content"]
        max_lcs_b = 0
        max_lcs_u = 0
        max_edit = 0
        for trainContent in contentList_train:
            LCS_B, LCS_U, Score_Edit = Name_Similarity(content, trainContent)
            if(LCS_B>max_lcs_b):
                max_lcs_b = LCS_B
            if(LCS_U>max_lcs_u):
                max_lcs_u = LCS_U
            if(Score_Edit>max_edit):
                max_edit = Score_Edit
            #print(LCS_B, LCS_U, Score_Edit)
        LCS_B_scores.append(max_lcs_b)
        LCS_U_scores.append(max_lcs_u)
        Edit_scores.append(max_edit)
        k+=1
        if(max_lcs_b>threshold):
            lcsu += 1
        if(max_lcs_u>threshold):
            lcsb += 1
        if(max_edit>threshold):
            editscore += 1
        
        if(k%10 == 0):
            print("{}/{}: LCS_U:{}, LCS_B:{}, Edit:{}".format(k, total_test, lcsu/k, lcsb/k, editscore/k))

    print(lcsu/total_test)
    print(lcsb/total_test)
    print(editscore/total_test)
    trainData.close()
    testData.close()