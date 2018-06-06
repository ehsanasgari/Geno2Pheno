__copyright__ = "Copyright 2017-2018, HH-HZI"
__author__ = "Ehsaneddin Asgari"
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Ehsaneddin Asgari"
__email__ = "asgari@berkeley.edu ehsaneddin.asgari@helmholtz-hzi.de"


import sys

sys.path.append('../')
from data_access.data_access_utility import ABRDataAccess
from classifier.classical_classifiers import SVM, RFClassifier, KNN
from chi2analysis.chi2analysis import Chi2Analysis
from phylochi2.phylochi2 import PhyloChi2

feature_lists=[['snps_nonsyn_trimmed']]#,['gpa_trimmed','gpa_roary'],['genexp_percent']

ABRAccess=ABRDataAccess('/net/sgi/metagenomics/projects/pseudo_genomics/results/amr_toolkit/intermediate_reps/',feature_list)
for drug in ABRAccess.BasicDataObj.drugs:
  print(drug , ' features ',' and '.join(feature_list))
  X_rep, Y, features, final_isolates = ABRAccess.get_xy_prediction_mats(drug, mapping={'0':0,'0.0':0,'1':1,'1.0':1})
  nwk='(((((MHH17247:0.050401925,(ZG302451:0.002248686,(MHH0985:0.005587828,((CH4411:0.002725400,PSAE1934:0.002113896)1.000:0.001407860,((M70640096:0.003609963,(((CH2682:0.003893431,MHH15187:0.003575611)1.000:0.003794279,CH2527:0.002852878)0.870:0.000381403,(F1800:0.001825937,CH4733:0.005472771)1.000:0.001296710)0.805:0.000413262)0.998:0.000561990,(RefCln_C-NN2:0.009157155,CH4549:0.001510928)1.000:0.001349550)1.000:0.001438653)1.000:0.002899182)1.000:0.013467973)1.000:0.036716865)0.985:0.004460714,((CH5548:0.028641560,(MHH15817:0.007155240,(F2176:0.003777449,(((F1821:0.000662041,F1789:0.001907441)0.999:0.000225470,(ZG5051896:0.001966570,F2054:0.000983848)1.000:0.001043727)0.170:0.000060170,((F1812:0.001987321,F1760:0.000838217)0.349:0.000247741,F1752:0.000955439)0.986:0.000156335)1.000:0.004453490)1.000:0.006608335)1.000:0.025199146)1.000:0.027504561,(CH5638:0.015469222,ZG302432:0.002350229)1.000:0.034938594)1.000:0.006024576)1.000:0.002526865,((MHH17783:0.034849221,((F2005:0.001516775,F1997:0.001256518)1.000:0.003580874,(CH4703:0.005436539,M70646050:0.003827268)1.000:0.000548819)1.000:0.032111360)1.000:0.005925246,((ESP038:0.008930812,MHH17546:0.009461628)1.000:0.033681992,(CH5356:0.037310680,M70563004:0.036862743)1.000:0.005867093)1.000:0.002204551)0.829:0.001590279)0.975:0.001865455,((CH5353:0.036424367,(CH5262:0.021686909,(RefCln_DK2:0.013873651,ESP035:0.004980580)1.000:0.023002185)1.000:0.011983863)0.999:0.006327360,(((CH2678:0.000638742,MHH16050:0.000578848)1.000:0.033038006,(((CH2724:0.003527561,(MS5:0.000508149,MS2:0.000707981)1.000:0.004179905)1.000:0.000901552,((CH4840:0.009967141,(ZG5003493:0.008614874,(CH2502:0.002070568,ZG302368:0.002486028)1.000:0.006023152)1.000:0.003424801)1.000:0.003082606,CH2848:0.004557876)1.000:0.000459968)1.000:0.005253160,(ZG302367:0.012336471,M70652746:0.005813567)1.000:0.004986769)1.000:0.025932333)1.000:0.006670643,((M70639645:0.002197074,(ZG316716:0.011047362,(ZG314777:0.004835965,ZG8038581181:0.003208789)0.917:0.001103552)1.000:0.004996018)1.000:0.034011322,(MHH0147:0.002926445,F1764:0.001645008)1.000:0.037694952)0.996:0.004766926)0.807:0.002291578)1.000:0.005577597)1.000:0.004236071,(((((CH2756:0.042276044,F1758:0.044307742)1.000:0.008672534,(((CH2747:0.001629838,(PSAE1975:0.000590248,PSAE2125:0.000754267)1.000:0.003075640)1.000:0.001411738,((MS1:0.006554337,(M70656672:0.008094605,CH4443:0.004219764)1.000:0.001311069)1.000:0.003644132,(M70641593:0.002220510,((M70649365:0.004825107,((F1823:0.002373919,(M70608424:0.000589699,M70604538:0.000589797)1.000:0.005094333)1.000:0.005564705,ZG302431:0.003481171)0.985:0.000759391)1.000:0.002585656,((((F2137:0.000691625,F2148:0.000723107)1.000:0.000201468,((F2165:0.000533223,F2166:0.000792975)1.000:0.000234516,F2234:0.000748468)0.880:0.000174922)1.000:0.000322736,((F2006:0.001133644,F2119:0.007855360)0.523:0.000169569,(F2035:0.000377029,F2017:0.001023674)0.999:0.000231847)1.000:0.000406633)1.000:0.002605548,(CH2875:0.002427238,CH4990:0.001597854)0.739:0.000692807)1.000:0.002211789)1.000:0.001289998)1.000:0.003802619)0.987:0.000210012)1.000:0.035441855,(RefCln_SCV20265:0.004010322,ESP006:0.005173521)1.000:0.042085394)1.000:0.005013622)1.000:0.003864214,((((ESP011:0.008177489,(MHH0426:0.010928287,((ESP064:0.000729586,(ESP084:0.001055462,ESP069:0.000872322)0.961:0.000248662)0.852:0.000208895,ESP078:0.000479415)1.000:0.008716000)1.000:0.002247155)1.000:0.012376121,((CF609_Iso3:0.001221158,ESP024:0.000702419)1.000:0.008030060,(MHH14929:0.007566624,(PSAE1984:0.015379648,ESP060:0.006361691)1.000:0.004495708)1.000:0.002349741)1.000:0.009115120)1.000:0.027397548,(((F1745:0.002503158,((F2240:0.000315505,F2199:0.000455499)1.000:0.001925415,(((F2059:0.000424474,F2055:0.000436885)0.915:0.000096490,(F1796:0.000567508,F1798:0.000850049)1.000:0.000316044)1.000:0.000408294,(F1712:0.000672418,(((ESP079:0.000471552,(((ESP088:0.000780800,ESP074:0.000427867)0.983:0.000141460,ESP082:0.000436579)0.999:0.000104163,(ESP061:0.000308967,ESP043:0.000827616)1.000:0.000142367)0.996:0.000100115)0.896:0.000063661,(ESP083:0.000362914,ESP063:0.000609793)0.436:0.000133754)0.990:0.000082158,(ESP066:0.000437261,(ESP047:0.000882449,ESP071:0.000939317)0.013:0.000101752)1.000:0.000126351)1.000:0.000301161)0.279:0.000084030)1.000:0.000582762)1.000:0.000469370)1.000:0.001513326,(F2235:0.003864218,(MHH15083:0.000713007,MHH15275:0.000530254)1.000:0.001288778)1.000:0.000565628)1.000:0.018911086,(CH4520:0.000966497,CH4785:0.000574781)1.000:0.014185262)1.000:0.025238225)1.000:0.007991358,((MHH15015:0.009437635,(PSAE1970:0.005998441,(F2034:0.000576286,(F1880:0.001383987,F1883:0.000444968)0.999:0.000325669)1.000:0.004828374)1.000:0.004687924)1.000:0.036177492,ESP055:0.041467650)1.000:0.006815111)1.000:0.003333432)1.000:0.005217487,(((CH2735:0.006199230,(MHH15280:0.003565543,CH4528:0.003505469)0.198:0.001913995)1.000:0.038968557,(CH5531:0.021482564,(ESP057:0.000368143,ESP059:0.000300494)1.000:0.018733503)1.000:0.021727715)1.000:0.003899302,(CH4704:0.034233661,ZG205710:0.039019908)1.000:0.007775279)1.000:0.002002517)1.000:0.002305160,(((((ESP073:0.000547875,ESP072:0.002058719)1.000:0.048369201,(F1727:0.015497369,((F2236:0.003049974,(F1775:0.003707527,CH5363:0.004858123)1.000:0.001553344)1.000:0.021873115,((CH2598:0.006282877,CH2707:0.008537924)1.000:0.012967788,(ESP081:0.023799907,ESP029:0.003763336)1.000:0.006486279)1.000:0.008169451)1.000:0.006048029)1.000:0.023682580)1.000:0.006121394,(CH2939:0.042935089,MHH17767:0.037682212)1.000:0.010188262)1.000:0.003922385,(((F1968:0.030125547,CH5362:0.036687258)1.000:0.010614227,(((CH2718:0.015525372,ESP058:0.005852480)1.000:0.019536020,((CH4681:0.004295031,(CH3466:0.003818981,ESP022:0.001309555)1.000:0.002589283)1.000:0.004726948,((((((ZG302359:0.010792278,CH3882:0.012022737)1.000:0.002809626,(F2230:0.004688084,(F2010:0.001951680,ZG314696:0.002777368)0.999:0.000449046)1.000:0.003491443)1.000:0.003049892,((F1706:0.000425834,F1748:0.000674340)1.000:0.000361904,(F2093:0.000552695,F2065:0.000467937)1.000:0.001143905)1.000:0.000620184)1.000:0.002022774,(F2128:0.000507232,(F2081:0.000418947,F2084:0.000426535)0.504:0.000109312)1.000:0.002177744)1.000:0.002626753,(F1787:0.003779673,CH3570:0.032535766)1.000:0.001826526)0.997:0.000531950,(CH4591:0.008132316,F2205:0.003580512)1.000:0.001078134)1.000:0.005906996)1.000:0.015875863)1.000:0.013631162,(MHH14990:0.026040324,RefCln_PAO1:0.025223377)1.000:0.011871058)1.000:0.004560791)1.000:0.005501517,CH3177:0.046110209)1.000:0.003372859)1.000:0.001372300,((CH5621:0.041040837,((CH3549:0.033085607,(PSAE1439:0.000243042,(PSAE1401:0.000581775,PSAE1438:0.000214701)0.993:0.000243853)1.000:0.038151119)1.000:0.006049818,((F1763:0.003414828,(F1757:0.001379825,(CH5551:0.001082643,CH2623:0.001256407)1.000:0.000792626)0.991:0.000706093)1.000:0.039164624,(PSAE1742:0.019685819,((M70646077:0.001696675,(CH4914:0.008045307,CH2698:0.006040241)1.000:0.003530722)0.543:0.001438504,(ZG205864:0.009813934,(ZG322541:0.002437487,(CH3613:0.002908188,(F2098:0.006606881,(PSAE1745:0.005636301,(ZG5089456:0.007003741,(ZG8510487:0.000800955,ZG8525123:0.000915266)1.000:0.003326503)1.000:0.003922392)1.000:0.002528281)0.956:0.000668831)1.000:0.001830381)1.000:0.001225785)1.000:0.002129247)1.000:0.013073474)1.000:0.034947249)1.000:0.009417035)1.000:0.004118342)1.000:0.007089239,((CH3325:0.033313078,CH2706:0.028586790)1.000:0.016901410,(ZG205565:0.029515400,(M70565254:0.006213401,((ZG302370:0.001728825,PSAE1872:0.001102667)1.000:0.005447457,(PSAE1630:0.002006828,ZG205694:0.002614271)1.000:0.006454814)1.000:0.001720531)1.000:0.037663705)1.000:0.011276604)1.000:0.002946500)1.000:0.005007723)1.000:0.003326651)0.579:0.001220199,(((((CH5066:0.022344354,F1894:0.023769046)1.000:0.019664157,(PSAE1912:0.036950971,(F2029:0.034828551,((MHH16371:0.000849614,MHH17501:0.000466195)1.000:0.021771983,(CH5182:0.003537998,CH5597:0.008344968)1.000:0.015518609)1.000:0.012512964)1.000:0.006869453)1.000:0.004711815)0.983:0.002547270,((ESP044:0.037895658,M70647319:0.032301316)1.000:0.007158885,(ESP067:0.026221165,MHH16459:0.029242706)1.000:0.009948338)1.000:0.002877972)1.000:0.002165556,(((MHH2419:0.006056787,(M70565897:0.002161786,((CH4891:0.000990267,(PSAE1903:0.000983510,(CH3906:0.000866035,CH3484:0.003038891)1.000:0.000567856)1.000:0.000540388)1.000:0.004257522,(CH4992:0.004186951,(CH4755:0.001003261,(CH4860:0.000405727,CH2582:0.003529662)0.944:0.000364345)1.000:0.000648989)1.000:0.004099774)1.000:0.003764926)1.000:0.005965527)1.000:0.034431494,((CH4083:0.000859971,CH4584:0.000312091)1.000:0.036712343,(CH2680:0.008463238,(CH2690:0.007070801,(MHH16427:0.000260869,(CH5478:0.000903151,MHH16208:0.000360359)0.063:0.000200110)1.000:0.015398421)1.000:0.007515396)1.000:0.022607270)1.000:0.014364464)1.000:0.004824999,(CH2597:0.033431698,(ESP009:0.000419653,ESP025:0.001054145)1.000:0.034314241)1.000:0.007197570)1.000:0.002963918)1.000:0.001930727,(((MHH15151:0.035643948,((ESP036:0.000940260,(ZG02420619:0.001031833,((MHH16798:0.000384656,(MHH16610:0.000547112,MHH16513:0.000465823)1.000:0.000664728)0.528:0.000246415,MHH16379:0.000228934)1.000:0.006322590)1.000:0.007558532)1.000:0.032700740,((F1688:0.000707812,F1689:0.000495181)1.000:0.004684549,(CH2860:0.009173885,CH2660:0.000722953)1.000:0.005143090)1.000:0.026752663)1.000:0.007341629)1.000:0.003619310,(((CH4878:0.025818432,CH4446:0.026689103)1.000:0.009746597,(F2075:0.028552559,RefCln_LESB58:0.030272298)1.000:0.004678295)1.000:0.002853403,(ZG316694:0.030865611,F1759:0.038429389)1.000:0.004921214)1.000:0.004798563)1.000:0.002398062,(((RefCln_PAK:0.041814254,(ESP027:0.009291626,(F2172:0.006594513,(CH4438:0.001643477,(CH4509:0.000551002,CH2522:0.000923005)1.000:0.000491861)1.000:0.011863640)1.000:0.016820629)1.000:0.026264248)1.000:0.004733157,((ESP023:0.033597136,(PSAE2325:0.000759997,PSAE2126:0.000276907)1.000:0.041767502)1.000:0.007164954,((ESP013:0.007635303,ESP012:0.000230393)1.000:0.050346411,(CH4780:0.064279851,((CH5387:0.013683125,(CH5206:0.001908639,CH4684:0.009648115)1.000:0.042048918)1.000:0.182876163,((ESP039:0.000575047,ESP040:0.000453341)1.000:0.069544742,(ESP034:0.042708104,(((CH5528:0.012885888,(MHH16951:0.001998172,M70564993:0.000776205)1.000:0.007963922)1.000:0.027980105,ESP031:0.062266358)1.000:0.023954003,((ZG316722:0.047500779,CH4862:0.044640891)1.000:0.010281751,((((CH2685:0.005099474,(PSAE1641:0.000430121,PSAE1645:0.000291808)1.000:0.005558794)1.000:0.053320211,(((F2044:0.003199940,(M70663247:0.004480916,CH2687:0.003651561)1.000:0.000577127)1.000:0.054914909,((CH2677:0.006886765,CF592_Iso2:0.007412720)1.000:0.001729656,((PSAE1635:0.000624598,(((PSAE1654:0.000627673,PSAE1837:0.000633345)1.000:0.000484033,(PSAE1649:0.000627586,(PSAE1647:0.000642059,(PSAE1651:0.000574369,PSAE1656:0.000804554)1.000:0.000261453)0.297:0.000125413)0.989:0.000146912)0.724:0.000134484,PSAE1642:0.000590482)1.000:0.000513590)1.000:0.007685406,(F1862:0.001070009,ZG301975:0.001034233)1.000:0.009923257)1.000:0.004816822)1.000:0.052011533)1.000:0.012151624,((((CH5596:0.031649243,ESP028:0.026930214)1.000:0.024143747,(CH4489:0.040268473,((((MHH15204:0.001859563,(CH2713:0.000497413,(ZG02488718:0.000075406,(RefCln_UCBPP-PA14:0.002354813,ZG02512057:0.000152273)1.000:0.001135122)1.000:0.000774901)0.517:0.000012556)1.000:0.000366785,(CH3462:0.003474524,(ZG302442:0.006271595,CH2500:0.006222144)1.000:0.001894695)1.000:0.002021097)1.000:0.000119602,MHH1883:0.000376497)1.000:0.001383710,((CH2829:0.012282089,((F2003:0.000522331,F2020:0.000182669)1.000:0.000640238,(CH4757:0.001351329,ZG314738:0.001268557)1.000:0.000437547)1.000:0.002094550)0.896:0.001020479,ESP021:0.004043477)1.000:0.002620831)1.000:0.037170620)1.000:0.016792364)0.657:0.003201315,(((F2045:0.000538658,(F1754:0.001075977,MHH1525:0.001675353)1.000:0.005030865)1.000:0.002890568,(ESP020:0.001499607,(ZG5048010:0.000898962,ZG8006959:0.000967087)1.000:0.019331067)1.000:0.000929513)1.000:0.049458516,((CH5334:0.003126678,(MHH16727:0.000521889,(MHH14911:0.000976498,MHH15103:0.000587973)0.205:0.000292634)1.000:0.002277357)1.000:0.046264654,((CH4560:0.006308039,CH5174:0.002293742)1.000:0.003301320,(CH2608:0.001247668,((PSAE2127:0.003832063,((CH4418:0.012292266,CH5052:0.004475587)1.000:0.004732754,(((((F2208:0.001347324,(CH3290:0.001613983,F1853:0.001159033)1.000:0.000695103)0.928:0.000534946,((F1724:0.000686319,((((F2056:0.000543619,F2040:0.000705686)0.961:0.000204186,(F1795:0.001591779,F2064:0.000863758)1.000:0.000345283)1.000:0.000178072,(F1741:0.000755615,F1810:0.000788701)0.963:0.000154930)1.000:0.000177935,(F2030:0.000629448,(((F2000:0.000670369,F1864:0.000619245)1.000:0.000275741,(F1697:0.000868266,(((F2167:0.000746896,F2097:0.001413003)0.995:0.000288063,F1746:0.000543691)0.477:0.000141457,F1747:0.000527719)1.000:0.000278703)0.827:0.000111548)1.000:0.000331409,(F1979:0.000689307,F1983:0.000518537)1.000:0.000543278)1.000:0.000400533)1.000:0.000261829)0.096:0.000054096)1.000:0.000287002,F1869:0.002731772)1.000:0.000631609)1.000:0.001380252,((M70638412:0.000553119,(M70635118:0.003030083,Vb20477:0.002584562)0.969:0.000288393)1.000:0.004530321,(ESP068:0.000661333,ESP091:0.000852291)1.000:0.009491105)1.000:0.002303315)1.000:0.003717483,((((((CH4548:0.000563769,CH2543:0.001078425)1.000:0.000211173,(((CH4745:0.000613503,CH2639:0.000671965)1.000:0.000235130,((CH5193:0.000464582,(CH4035:0.000713985,CH2591:0.000685097)0.852:0.000158990)0.847:0.000080715,(CH2657:0.000869316,CH3173:0.001644336)0.969:0.000216605)0.815:0.000081907)0.996:0.000170125,(((CH5695:0.000609416,CH5666:0.000625959)1.000:0.000432984,CH5464:0.000743571)1.000:0.000283761,CH2560:0.000633578)0.981:0.000172148)0.929:0.000093832)0.979:0.000142803,(CH4602:0.000819698,(CH4916:0.000897397,CH5159:0.002087782)0.277:0.000322279)1.000:0.000665491)1.000:0.003384348,(CH3648:0.000811751,CH5022:0.005902928)1.000:0.002640462)1.000:0.000883234,(CH2675:0.000575629,CH2699:0.000813898)1.000:0.001656721)1.000:0.003185594,PSAE2139:0.009224965)1.000:0.001879140)0.918:0.000321529,(CH5688:0.003554419,CH2748:0.005009000)1.000:0.004314552)1.000:0.003035547)1.000:0.002953832)1.000:0.003427375,((CH2674:0.001425810,MHH17233:0.001303937)1.000:0.000958122,(MHH17441:0.001112544,ESP053:0.000665100)0.420:0.000155864)0.999:0.000155176)1.000:0.001138549)1.000:0.000763079)1.000:0.045592698)0.746:0.008452754)1.000:0.007833020)1.000:0.005829502,(((((ZG316704:0.000848423,(MHH16530:0.000440091,MHH16563:0.000411413)1.000:0.000613073)1.000:0.003163406,(F1659:0.000198333,(F1957:0.000927016,F1801:0.000561984)1.000:0.003805629)1.000:0.002954905)1.000:0.001119712,(CH4634:0.004769378,(CH2672:0.001908250,(CH5267:0.000926712,ESP070:0.001260436)0.997:0.000301885)1.000:0.004303313)1.000:0.001673373)1.000:0.055187129,(CH2734:0.016195375,(ESP037:0.005178535,CH2730:0.005454294)1.000:0.003484276)1.000:0.040415431)1.000:0.008273927,(CH3797:0.040973559,(F2073:0.002587201,(CH5591:0.003674691,((ZG5021922:0.006123796,(CH4877:0.002374082,CH2705:0.000832533)0.671:0.000315421)1.000:0.006443111,CH2665:0.001933470)1.000:0.000981029)1.000:0.003224700)1.000:0.048256436)1.000:0.011687541)1.000:0.004502882)1.000:0.003850381)0.829:0.004031837)0.987:0.004430548,((ESP003:0.000689431,ESP004:0.000596435)1.000:0.058753528,(ESP002:0.050418651,(CH2824:0.053410768,((ESP046:0.044751409,(Vb3320:0.034900842,(CH2922:0.059353103,(ESP076:0.000596608,ESP075:0.000489491)1.000:0.058108219)1.000:0.014114828)1.000:0.011773409)1.000:0.016175846,((ESP050:0.006115944,(ESP033:0.001051015,ESP015:0.001048061)1.000:0.007293945)1.000:0.043573725,(ZG302383:0.018559583,CH5462:0.008433596)1.000:0.044539531)1.000:0.007976355)1.000:0.009918916)1.000:0.010663987)1.000:0.012457742)1.000:0.017603068)0.999:0.004866865,(RefCln_PA7:0.000000006,(CH4433:0.020573629,ESP077:0.022453610)1.000:0.127823492)1.000:2.723492228)0.987:0.000670793)1.000:0.008629544)1.000:0.038106900)0.869:0.015028639)1.000:0.010910062)1.000:0.007849717)1.000:0.012232250)1.000:0.006662997)1.000:0.007135304)0.998:0.002173268,((CH4766:0.042498612,(CH5550:0.027894761,MHH0927:0.024626713)1.000:0.009390003)1.000:0.003758157,((CH5432:0.046200943,(CH5291:0.002261419,CH2764:0.008544311)1.000:0.041826942)1.000:0.012340881,(ESP032:0.047048867,CH2658:0.034631341)0.864:0.005656964)1.000:0.004669786)1.000:0.002247983)1.000:0.001618281)1.000:0.003650526)1.000:0.002385302);'
  PhyloChi2=PhyloChi2(nwk, X_rep, Y, features, final_isolates)
  PCHI2.generate_parallel_gainloss_data_for_drug(10)