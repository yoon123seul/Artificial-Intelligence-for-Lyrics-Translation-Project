#전과정: 파이썬 pip, 자바 설치/환경 변수 설정 후,
#pip install konlpy


from konlpy.tag import Kkma     #KKma: 형태소 분석 모듈
from jamo import h2j,j2h,j2hcj  #초성,중성,종성 분리,조합
import random
import numpy as np
import Rulebase_1 as KH
import Rulebase_2 as MS
import Rulebase_3 as ML


kkm = Kkma()
special_syl=[',' , '.' , '!' , '?' , '/' , '(' , ')', '~' , '_' , ':' , '"']
Diphthong=['ㅑ','ㅒ','ㅕ','ㅖ','ㅘ','ㅙ','ㅛ','ㅝ','ㅞ','ㅠ','ㅢ']
initial_sound=['ㄱ','ㄴ','ㄷ','ㄹ','ㅁ','ㅂ','ㅅ','ㅇ','ㅈ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ','ㄲ','ㄸ','ㅃ','ㅆ','ㅉ']
middle_sound=['ㅏ','ㅑ','ㅓ','ㅕ','ㅗ','ㅛ','ㅜ','ㅠ','ㅡ','ㅣ','ㅐ','ㅔ','ㅚ','ㅟ','ㅙ','ㅞ','ㅒ','ㅖ','ㅢ','ㅘ','ㅝ']

tryofresult=3 #높을수록 점수의 비중이 커짐.
numofresult=100 #결과 시도하는 개수 

# 입력값 형태: "미안하지만, 너를 사랑해.",8(목표 함수 길이)
def maketxt(txt0,req_syl):
    #txt0="미안하지만, 너를 사랑해"

    txt_syl=0
    txt1=""
    # 글자수 세는 과정
    for i in txt0:
        if   i=='+': txt_syl=txt_syl+1
        elif i=='-': txt_syl=txt_syl-1
        elif i==' ': txt1=txt1+i
        elif i in special_syl: pass
        else:
            txt1 = txt1 + i
            txt_syl = txt_syl + 1
    sub_syl = req_syl - txt_syl
    #txt1="미안하지만 너를 사랑해"

    if sub_syl>=0: #늘려야 하는 경우 -발음을 늘린다
        sylc=[0 for i in range(len(txt1))]
        sylp=[0 for i in range(len(txt1))]
        sylccount=0
        sylpcount=0
        used=[]
        ans=["" for i in range (numofresult)]
        for i in range(len(txt1)): #이중모음의 개수를 찾아봄
            if txt1[i] != ' ':
                morp=KH.convert(txt1[i])
                if morp[1] in Diphthong:
                    sylc[i]=1
                    sylccount=sylccount+1
        for i in range(10): #10개의 결과값을 제시-랜덤으로 제시, 같을 수 있음
            while sub_syl>=txt_syl: #늘려야 하는 음절의 수가 전체 음절의 수보다 많은 경우-거의 없는 경우, 모든 음절을 늘린 뒤 추가로 늘림
                sylp=[sylp[j]+1 for j in range(len(txt1))]
                sub_syl=sub_syl-txt_syl
            if sylccount>=sub_syl: #이중 모음의 개수가 늘려야 할 발음보다 많은 경우-이중 모음 중에서만 발음을 늘림
                while sylpcount<sub_syl:
                    k=random.randrange(0,len(sylc))
                    if sylc[k]==1:
                        if k not in used:
                            used.append(k)
                            sylp[k]=sylp[k]+1
                            sylpcount=sylpcount+1
            else:                 #이중 모음의 개수가 늘려야 할 발음보다 적은 경우-이중 모음에서 발음을 모두 늘리고 추가로 늘릴 발음을 찾음
                while sylpcount<sylccount:
                    k=random.randrange(0,len(sylc))
                    if sylc[k]==1:
                        if k not in used:
                            used.append(k)
                            sylp[k]=sylp[k]+1
                            sylpcount=sylpcount+1    
                while sylpcount<sub_syl:
                    k=random.randrange(0,len(sylc))
                    if k not in used and txt1[k]!=' ':
                        used.append(k)
                        sylp[k]=sylp[k]+1
                        sylpcount=sylpcount+1
            for j in range(len(txt1)): #늘리는 경우 결과값 string을 만듦-늘여야 할 발음은 ~로 만듦
                ans[i]=ans[i]+txt1[j]
                ans[i]=ans[i]+"~" * sylp[j]
        return ans #늘리는 경우 여기까지
                          
                    






    #줄여야 하는 경우
    txt2=kkm.pos(txt1,flatten=False, join=True) #kkma
    #txt2=[['미안/NNG', '하/XSV', '지만/ECE'], ['너/NP', '를/JKO'], ['사랑/NNG', '하/XSV', '어/ECS']]
    syld=[[0 for j in range(len(txt2[i]))] for i in range(len(txt2))]

    #데이터 저장 작업
    A=[]    #사랑                                   #를
    B=[]    #['ㅅ', 'ㅏ', '#', 'ㄹ', 'ㅏ', 'ㅇ']    #['ㄹ', 'ㅡ', 'ㄹ']
    C=[]    #NNG                                   #JKO        
    D=[]    #0(점수)                               #-1
    E=[]    #2(음절)                               #1
    F=[]    #[]                                    #[['ㄹ',1],['',1]]
                
    for i in txt2:
        for j in i:
            for k in range(len(j)):
                if j[k]=='/':
                    typ=j[k+1:]
                    cnt=j[:k]
            morp_score=MS.score[typ]
            if cnt[0] in initial_sound: #'ㅂ니다'같은 경우
                morp=['#','#',cnt[0]]
                if KH.convert(cnt[1:])!=[]: morp.append(KH.convert(cnt[1:]))
                clen=len(cnt)-1
            elif cnt[0] in middle_sound: #'ㅏㅆ었'같은 경우
                if cnt[1] in initial_sound:
                    morp=['#',cnt[0],cnt[1]]
                    if KH.convert(cnt[2:])!=[]: morp.append(KH.convert(cnt[2:]))
                    clen=len(cnt)-2
                else:
                    morp=['#',cnt[0],'#']
                    if KH.convert(cnt[1:])!=[]: morp.append(KH.convert(cnt[1:]))
                    clen=len(cnt)-1
            else:
                morp=KH.convert(cnt)
                clen=len(cnt)
            #print(cnt)              #사랑
            #print(morp)             #['ㅅ', 'ㅏ', '#', 'ㄹ', 'ㅏ', 'ㅇ']
            #print(typ)              #NNG
            #print(morp_score)       #0

            A.append(cnt)
            B.append(morp)
            C.append(typ)
            D.append(morp_score)
            E.append(clen)

        A.append('')
        B.append(['#','#','#'])
        C.append('BLK')
        D.append(0)
        E.append(0)
    F=[[] for i in A]

    #print(A)    #사랑
    #print(B)    #['ㅅ', 'ㅏ', '#', 'ㄹ', 'ㅏ', 'ㅇ']
    #print(C)    #NNG
    #print(D)    #0(morp점수)
    #print(E)    #음절수
    #print(F)    #바꿀 후보 
    txt_syl=np.sum(E)
    sub_syl=req_syl-txt_syl
    


    #우선순위별로 줄이기
    def getreducetxt():
        copyD=D[:]
        req_reduce_syl=-sub_syl
        can_reduce_syl=0
        F=[[] for i in A]
        G=[0 for i in A]
        max_reduce_syl=0
        def reducebyminus1(typelist,i):#-1점에서, 타입을 알때 그 타입에 맞는 F를 추가
            max_reduce_syl=0
            
            for x in typelist:

                if A[i] in x[0]:
                    wlen=E[i]
                    for y in x[1]:
                        if wlen>y[1]:
                            max_reduce_syl=max(wlen-y[1],max_reduce_syl)
                            F[i].append([y[0],wlen-y[1]])
                    return max_reduce_syl
            return 0

        for i in range(len(A)): #3점의 경우
            if copyD[i]==3:
                F[i].append(['',E[i]])
                G[i]=E[i]
                 

        for i in range(len(A)): #ㅇ발음의 탈락
            for j in range(0,len(B[i]),3):
                if B[i][j]=='ㅇ':
                    if j!=0 and B[i][j-1]=='#':
                        F[i].append([A[i]+'-',1])
                        G[i]=1
                    elif j==0 and i!=0 and B[i-1][-1]=='#':
                        k=i-1
                        while k>0 and B[k][-2]=='#':
                            k=k-1
                        if B[k][-2]!='#' and B[i][-1]=='#':
                            F[i].append([A[i]+'-',1])
                            G[i]=1 

        
        for i in range(len(A)): #-1점의 경우
            if copyD[i]==-1:
                if C[i]=='NR':#수사의 경우
                    G[i]=reducebyminus1(ML.NR,i)
                    copyD[i]=1 #변형한 후에는, submorpscore인 1로 바꿈. 만약 1점짜리 단어가 삭제될 때 수사 역시 사라질 수 있음
                if C[i]=='VCP':#긍정지정사의 경우
                    G[i]=reducebyminus1(ML.VCP,i)
                    copyD[i]=0 
                if C[i]=='VCN':#부정지정사의 경우
                    G[i]=reducebyminus1(ML.VCN,i)
                    copyD[i]=0 
                if C[i]=='MAC':#접속 부사의 경우
                    G[i]=reducebyminus1(ML.MAC,i)
                    copyD[i]=2 
                if C[i]=='JKS':#주격 조사의 경우
                    G[i]=reducebyminus1(ML.JKS,i)
                    copyD[i]=2 
                if C[i]=='JKO':#목적격 조사의 경우
                    G[i]=reducebyminus1(ML.JKO,i)
                    copyD[i]=2
                if C[i]=='JKM':#부사격 조사의 경우
                    G[i]=reducebyminus1(ML.JKM,i)
                    copyD[i]=1
                if C[i]=='JKI':#호격 조사의 경우
                    G[i]=reducebyminus1(ML.JKI,i)
                    copyD[i]=0
                if C[i]=='JKQ':#인용격 조사의 경우
                    G[i]=reducebyminus1(ML.JKQ,i)
                    copyD[i]=2
                if C[i]=='JX':#보조사의 경우
                    G[i]=reducebyminus1(ML.JX,i)
                    copyD[i]=1
                if C[i]=='EPT':#시제선어말어미의 경우
                    G[i]=reducebyminus1(ML.EPT,i)
                    copyD[i]=1
                if C[i] in['EFN','EFQ','EFO','EFA','EFI','EFR']:#종결어미의 경우
                    G[i]=0
                    if len(A[i])>=1 and A[i][-1] in ['요','고','구']:
                        G[i]=1
                        F[i].append([A[i][:-1],1])
                        for x in ML.EF:
                            if A[i][:-1] in x[0]:
                                wlen=E[i]-1
                                for y in x[1]:
                                    if wlen>y[1]:
                                        G[i]=max(wlen-y[1]+1,G[i])
                                        F[i].append([y[0],wlen-y[1]+1])
                    G[i]=max(reducebyminus1(ML.EF,i),G[i])
                    copyD[i]=0
                if C[i]=='ETD':#관형형전성어미의 경우
                    G[i]=reducebyminus1(ML.ETD,i)
                    copyD[i]=0       

        can_reduce_syl=sum(G)
        if can_reduce_syl>=req_reduce_syl:
            for j in range(tryofresult):
                reduce_syl=0
                copyA=A[:]
                changed=[False for x in range(0,len(F))]
                x=0
                while reduce_syl<req_reduce_syl and x<len(F):
                    k=random.randrange(0,len(F))
                    x=x+1
                    if changed[k]==False and len(F[k])!=0:
                        changed[k]=True
                        a=F[k][random.randrange(0,len(F[k]))]
                        copyA[k]=a[0]
                        reduce_syl+=a[1]
                if reduce_syl==req_reduce_syl:
                    return copyA   
                

        for i in range(len(A)): #2점의 경우
            if copyD[i]==2:
                F[i].append(['',E[i]])
                G[i]=E[i]
        can_reduce_syl=sum(G)
        if can_reduce_syl>=req_reduce_syl:
            for j in range(tryofresult):
                reduce_syl=0
                copyA=A[:]
                changed=[False for x in range(0,len(F))]
                x=0
                while reduce_syl<req_reduce_syl and x<len(F):
                    k=random.randrange(0,len(F))
                    x=x+1
                    if changed[k]==False and len(F[k])!=0:
                        changed[k]=True
                        a=F[k][random.randrange(0,len(F[k]))]
                        copyA[k]=a[0]
                        reduce_syl+=a[1]
                if reduce_syl==req_reduce_syl:
                    return copyA   

        
        for i in range(len(A)): #1점의 경우
            if copyD[i]==1:
                F[i].append(['',E[i]])
                G[i]=E[i]

        can_reduce_syl=sum(G)
        if can_reduce_syl>=req_reduce_syl:
            for j in range(tryofresult):
                reduce_syl=0
                copyA=A[:]
                changed=[False for x in range(0,len(F))]
                x=0
                while reduce_syl<req_reduce_syl and x<len(F):
                    k=random.randrange(0,len(F))
                    x=x+1
                    if changed[k]==False and len(F[k])!=0:
                        changed[k]=True
                        a=F[k][random.randrange(0,len(F[k]))]
                        copyA[k]=a[0]
                        reduce_syl+=a[1]
                if reduce_syl==req_reduce_syl:
                    return copyA   
        return("!!생성 실패!!")

    #스트링으로 바꾸는 과정
    def retredans() :
        anspiece=getreducetxt()
        if anspiece==("!!생성 실패!!"): return ""
        ans=""
        for i in range(len(A)):
            if C[i]=='BLK': 
                ans=ans+" "
            elif i>0 and anspiece[i-1]=='하' and anspiece[i]=='어-':
                ans=ans[:-1]
                ans=ans+'해'
            elif len(anspiece[i])>=1 and anspiece[i][0] in initial_sound:
                s=ans[-1]
                if s==' ': pass
                else: 
                    k=j2hcj(h2j(ans[-1]))
                    ans=ans[:-1]
                    if len(k)==2:
                        ans=ans+j2h(k[0],k[1],anspiece[i][0])
                if len(anspiece[i])>=2:
                    ans=ans+anspiece[i][1:]
            else:
                ans=ans+anspiece[i] 
                
        return ans

    k=[]       
    for i in range(numofresult):
        k.append(retredans())
    return k

def make(testtxt,len):
    a=maketxt(testtxt,len)
    ans=[]
    w=0
    for k in a:
        if k not in ans and k!="":
            ans.append(k)
            w=w+1
        if w>=10: break
    return ans
    
if __name__ == '__main__':
    testtxt= ''
    result=(make(testtxt,8))
    for i in result:
        print(i)


# testtxt = '날개를 활짝 펴고'
# result=(make(testtxt,6))
# for i in result:
#     print(i)    