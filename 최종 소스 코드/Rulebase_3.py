NR=[
    [['하나'],[['한',1],['일',1]]],
    [['다섯'],[['오',1]]],
    [['여섯'],[['육',1]]],
    [['일곱'],[['칠',1]]],
    [['여덟'],[['팔',1]]],
    [['아홉'],[['구',1]]],
    [['첫째'],[['첫',1]]],
    [['둘째'],[['두',1]]],
    [['셋째'],[['세',1]]],
    [['넷째'],[['네',1]]],
    [['다섯째'],[['다섯',2]]],
    [['여섯째'],[['여섯',2]]],
    [['일곱째'],[['일곱',2]]],
    [['여덟째'],[['여덟',2]]],
    [['아홉째'],[['아홉',2]]],
    [['열째'],[['열',1]]],
    [['여드레'],[['팔일',2]]],
    [['아흐레'],[['구일',2]]]
]
VCP=[
    [['이다'],[['임',1],['',0]]]
]
VCN=[
    [['아니다'],[['아님',2]]]
]
MAC=[#kkma데이터셋 기준 상위 100개 빈도수의 접속 부사
    [['그렇지만','그러나','하지만','허지만','그래도','그래두','그란디','그른데','그치만','다만','허나','헌데'],[['하지만',3],['지만',2],['만',1],['',0]]],
    [['그리고','그리구','그러고','그르구','그라고','글구'],[['글고',2],['고',1],['',0]]],
    [['그러다가','그러면서','그러므로','그리하여','이리하여','그러더니','그러다','그래서','따라서','그래','그서'],[['그래서',3],['그서',2],['서',1],['',0]]],
    [['그런데','그런디','근데','근디'],[['근데',2],['ㄴ데',1],['',0]]],
    [['그러면은','그렇다면','그러니깐','그르니까','그러면','그러구','그러믄','그면은','그믄','그럼','그면','근깐','긍까'],[['그럼',2],['금',1],['',0]]],
    [['그러니까는','그니까는','그러니까','그니까','그니깐','근까는','근까','그까','그깐'],[['그니까',3],['근까',2],['',0]]],
    [['아니면은','아니면','아님','또는','혹은','또'],[['아니면',3],['아님',2],['또',1],['',0]]],
    [['왜나하면은','왜냐면은','왜나하면','왜냐면'],[['왜냐면',3],['왜냐',2],['',0]]],
    [['말하자면','이를테면','이른바','예컨대'],[['이른바',3],['',0]]],
    [['어쨌든','하여튼','허기야','한편','하튼'],[['한편',2],['',0]]],
    [['더구나','더욱이'],[['',0]]],
    [['그러자'],[['',0]]],
    [['하물며','하기야','하기는','하여간','하긴','허긴'],[['하긴',2],['',0]]],
    [['그러니'],[['',0]]],
    [['혹여나','혹시'],[['혹시',2],['혹',1],['',0]]],
    [['그렇다고','그래야','요컨데','오히려','한데','소위','역시','및','단','즉','금','곧','까'],[['',0]]]
]
JKS=[
    [['이','가','서'],[['',1]]],
    [['에서','께서','이서'],[['이',1],['가',0],['',0]]]
]
JKO=[
    [['를'],[['ㄹ',0],['',0]]],
    [['을'],[['',0]]]
]
JKM=[#kkma데이터셋 기준 상위 60개 빈도수의 부사격조사 중 줄일 수 있는것
    [['으로','으루','으루다'],[['로',1],['',0]]],
    [['에서'],[['서',1],['',0]]],
    [['에게'],[['게',1],['',0]]],
    [['하고','이랑','허구'],[['랑',1],['',0]]],
    [['에다가'],[['에다',2]]],
    [['로서','로써','으로서','으로써'],[['서',1],['',0]]],
    [['처럼'],[['',0]]],
    [['보다'],[['',0]]],
    [['으로부터','에게서','한테서','서부터','에서부터','부터'],[['부터',2],['',0]]],
    [['맹키로','만큼'],[['만큼',2],['',0]]],
    [['한테','에다','같이','대로','더러','보고','마냥','헌테','마따나'],[['',0]]]
]
JKI=[
    [['야','아','아아','이야','아야'],[['야',1],['',0]]],
    [['이여','여','이시여'],[['여',1],['이여',2],['',0]]]
]
JKQ=[
    [['고','라고','이라고','하고','라구','이라구','라'],[['라고',2],['고',1],['',0]]]
]
JX=[#kkma데이터셋 기준 상위 60개 빈도수의 부사격조사 중 줄일 수 있는것
    [['든지','든가'],[['든',1],['',0]]],
    [['마는','만은'],[['만',1],['',0]]],
    [['뿐이'],[['뿐',1],['',0]]]
]
JX=[#kkma데이터셋 기준 상위 60개 빈도수의 보조사 중 줄일 수 있는것
    [['든지','든가'],[['든',1],['',0]]],
    [['마는','만은'],[['만',1],['',0]]],
    [['뿐이'],[['뿐',1],['',0]]]
]
EPT=[#kkma데이터셋 기준 상위 60개 빈도수의 보조사 중 줄일 수 있는것
    [['았었','았'],[['았',1],['았-',0],['',0]]],
    [['었었','었'],[['었',1],['었-',0],['',0]]],
    [['ㅕㅆ었'],[['ㅕㅆ',0],['',0]]],
    [['ㅏㅆ었'],[['ㅏㅆ',0],['',0]]],
    [['ㅓㅆ었'],[['ㅓㅆ',0],['',0]]]
]
EF=[#기본적으로 '고,구,요'가 있으면 빼는 코드 있어야함-뺀 경우/빼고 변형/안빼고 변형 있어야함
    #kkma데이터셋 기준 상위 300개 빈도수의 종결어미 중 줄일 수 있는것
    [['ㄴ다','ㅂ니다','ㄴ다고'],[['ㄴ다',1],['ㅁ',0],['어-',0]]],
    [['다','습니다','읍니다'],[['슴다',2],['ㅁ다',1],['어-',0]]],
    [['는다'],[['음',1]]],
    [['어'],[['ㅁ',0]]],
    [['죠'],[['죠',1]]],
    [['을까','는가','느냐','냐','냐','나요','읍니까'],[['나',1]]],
    [['ㅂ니까','','ㄹ까'],[['ㄹ까',1]]],
    [['거든요'],['거든',2]],
    [['아라'],[['ㅏ',0]]],
    [['ㅂ시다'],[['자',1]]],
    [['으니까'],[['으니',2]]],
    [['라고'],[['고',1]]],
    [['답니다'],[['단다',2]]],
    [['ㄴ답니다'],[['ㄴ단다',2]]],
    [['랍니다'],[['란다',2]]],
    [['는구나'],[['는군',2]]],
    [['더구나','더구먼'],[['더군',2]]],
    [['예','이에'],[['임',1]]],
    [['ㅂ니까'],[['',0]]],
    [['구먼','구만','구려'],[['군',1]]],
    [['노','마'],[['ㅁ',0]]],
    [['어서'],[['어',1]]],
    [['로구나'],[['구나',2]]],
    [['게나'],[['게',1]]],
    [['다면'],[['담',1]]],
    [['더냐'],[['던',1]]],
    [['려무나'],[['렴',1]]],
    [['이다'],[['',0]]],
    [['다는구나'],[['다는군',3]]]
]
ETD=[#kkma데이터셋 기준 상위 60개 빈도수의 관형형전성어미 중 줄일 수 있는것
    [['는','은'],[['ㄴ',0]]],
    [['을'],[['ㄹ',0]]],
    [['다는'],[['단',1]]],
    [['라는'],[['란',1]]],
    [['려는'],[['련',1]]],
    [['는다는'],[['는단',2],['는',1]]],
    [['느냐는'],[['느냔',2]]],
    [['으려는'],[['으련',2]]],
    [['냐는'],[['냔',1]]],
    [['ㄴ다는'],[['ㄴ단',1]]],
    [['리라는'],[['리란',2]]],
    [['으리라는'],[['으리란',3]]],
    [['더라는'],[['더란',2]]],
    [['으라는'],[['으란',2]]],
    [['ㄴ가라는'],[['ㄴ가란',2]]],
    [['다라는'],[['다란',2]]],
    [['노라는'],[['노란',2]]],
    [['는가라는'],[['는가란',3]]],
    [['대는'],[['댄',1]]],
    [['으냐는'],[['으냔',2]]],
    [['ㄹ려는'],[['ㄹ련',1]]],
    [['ㄴ대는'],[['ㄴ댄',1]]],
    [['ㄹ까라는'],[['ㄹ까란',2]]]
]