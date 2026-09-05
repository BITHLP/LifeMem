TOOLQA_EASY8 = """Question: How many extra minutes did the DL1575 flight take from ATL to MCO on 2022-01-12?
Thought 1: This is a question related to flights. We need to load the flights database.
Action 1: LoadDB[flights]
Observation 1: We have successfully loaded the flights database, including the following columns: FlightDate, Airline, Origin, Dest, Cancelled, Diverted, CRSDepTime, DepTime, DepDelayMinutes, DepDelay, ArrTime, ArrDelayMinutes, AirTime, CRSElapsedTime, ActualElapsedTime, Distance, Year, Quarter, Month, DayofMonth, DayOfWeek, Marketing_Airline_Network, Operated_or_Branded_Code_Share_Partners, DOT_ID_Marketing_Airline, IATA_Code_Marketing_Airline, Flight_Number_Marketing_Airline, Operating_Airline, DOT_ID_Operating_Airline, IATA_Code_Operating_Airline, Tail_Number, Flight_Number_Operating_Airline, OriginAirportID, OriginAirportSeqID, OriginCityMarketID, OriginCityName, OriginState, OriginStateFips, OriginStateName, OriginWac, DestAirportID, DestAirportSeqID, DestCityMarketID, DestCityName, DestState, DestStateFips, DestStateName, DestWac, DepDel15, DepartureDelayGroups, DepTimeBlk, TaxiOut, WheelsOff, WheelsOn, TaxiIn, CRSArrTime, ArrDelay, ArrDel15, ArrivalDelayGroups, ArrTimeBlk, DistanceGroup, DivAirportLandings
Thought 2: We need to filter the information related to the query.
Action 2: FilterDB[Flight_Number_Marketing_Airline=1575, FlightDate=2022-01-12, Origin=ATL, Dest=MCO]
Observation 2: We have successfully filtered the data (1 row).
Thought 3: We then need to know the departure delayed time.
Action 3: GetValue[DepDelay]
Observation 3: -7.0
Thought 4: We then need to know the arrival delayed time.
Action 4: GetValue[ArrDelay]
Observation 4: -17.0
Thought 5: To compute the extra minutes, we need to subtract the departure delayed time from the arrival delayed time.
Action 5: Calculate[(-17)-(-7)]
Observation 5: -10
Thought 6: After calculation, we know that the answer is -10.
Action 6: Finish[-10]

Question: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
Thought 1: Maybe I can write a python program to solve this problem.
Action 1: PythonInterpreter[# solution in Python:
def solution():
    # Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
    golf_balls_initial = 58
    golf_balls_lost_tuesday = 23
    golf_balls_lost_wednesday = 2
    golf_balls_left = golf_balls_initial - golf_balls_lost_tuesday - golf_balls_lost_wednesday
    result = golf_balls_left
    return result]
Observation 1: 33
Thought 2: After execution of Python Code, we know that the answer is 33.
Action 2: Finish[33]

Question: What is the corresponding Mean_IoU score of the FRRN method on Cityscapes dataset for Semantic_Segmentation task?
Thought 1: The question is asking some performance information about Mean_IoU score of the FRRN method on Cityscapes dataset for Semantic_Segmentation task, we need to retrieve some useful information from the scirex database.
Action 1: RetrieveScirex[Mean_IoU score of the FRRN method on Cityscapes dataset for Semantic_Segmentation task]
Observation 1: section : Abstract Semantic image segmentation is an essential component of modern autonomous driving systems , as an accurate understanding of the surrounding scene is crucial to navigation and action planning . Current state - of - the - art approaches in semantic image segmentation rely on pretrained networks that were initially developed for classifying images as a whole . While these networks exhibit outstanding recognition performance ( i.e. , what is visible ? ) , they lack localization accuracy ( i.e. , where precisely is something located ? ) . Therefore , additional processing steps have to be performed in order to obtain pixel - accurate segmentation masks at the full image resolution . To alleviate this problem we propose a novel ResNet - like architecture that exhibits strong localization and recognition performance . We combine multi - scale context with pixel - level accuracy by using two processing streams within our network : One stream carries information at the full image resolution , enabling precise adherence to segment boundaries . The other stream undergoes a sequence of pooling operations to obtain robust features for recognition . The two streams are coupled at the full image resolution using residuals . Without additional processing steps and without pretraining , our approach achieves an intersection - over - union score of 71.8 % on the Cityscapes dataset .
Thought 2: As the approach achieves an intersection - over - union score of 71.8 % on the Cityscapes dataset, the answer should be 71.8%.
Action 2: Finish[71.8%]

Question: When was the paper Learning the Principle of Least Action with Reinforcement Learning. published?
Thought 1: The question is asking some basic information of a dblp paper. We need to load the DBLP graph.
Action 1: LoadGraph[dblp]
Observation 1: DBLP graph is loaded.
Thought 2: The question is asking the published date of a paper, we need to check the node from the PaperNet in DBLP graph.
Action 2: NodeCheck[PaperNet, Learning the Principle of Least Action with Reinforcement Learning.]
Observation 2: {'title': 'Learning the Principle of Least Action with Reinforcement Learning.', 'year': 2021, 'venue': {'raw': 'AAAI Spring Symposium - MLPS'}, 'n_citation': 0, 'keywords': [], 'doc_type': 'Conference', 'page_start': '', 'page_end': ''}
Thought 3: The published date of the paper is 2021.
Action 3: Finish[2021]

Question: How many collaborators does Chao Zhang have in the DBLP graph?
Thought 1: The question is asking the number of collaborators of a person, we need load the DBLP graph.
Action 1: LoadGraph[dblp]
Observation 1: DBLP graph is loaded.
Thought 2: The question is asking the collaborators of a person, we need to check the neighboring nodes from the AuthorNet in DBLP graph.
Action 2: NeighbourCheck[AuthorNet, Chao Zhang]
Observation 2: ['YUHUI YUAN', 'Rao Fu', 'Lang Huang', 'Weihong Lin', 'X Chen', 'Jingdong Wang']
Thought 3: The number of collaborators of Chao Zhang is 6.
Action 3: Finish[6]

Question: How many papers does Chao Zhang and Weihong Lin have in common in the DBLP graph?
Thought 1: The question is asking the number of common papers of two persons, we need load the DBLP graph.
Action 1: LoadGraph[dblp]
Observation 1: DBLP graph is loaded.
Thought 2: The question is asking the common papers of two persons, we need to check the edges between them from the PaperNet in DBLP graph.
Action 2: EdgeCheck[PaperNet, Chao Zhang, Weihong Lin]
Observation 2: {'weight': 1, 'papers': ['HRFormer: High-Resolution Vision Transformer for Dense Predict.'], 'n_citation': [95]}
Thought 3: The number of common papers of Chao Zhang and Weihong Lin is 1.

Question: Where did Stephen's Opera performance take place?
Thought 1: The question is asking the location of Stephen's Opera performance from agenda.
Action 1: RetrieveAgenda[Stephen's Opera performance]
Observation 1: On January 29, 2022, there will be an opera performance at the Lyric Opera House, featuring Stephen. The show will start at 7:00 PM and end at 9:00 PM. It promises to be a wonderful evening of beautiful music and powerful performances in a stunning venue. Come and experience the magic of opera at its finest!
Thought 2: The event happened in Lyric Opera.
Action 2: Finish[Lyric Opera]

Question: What was the trading volume of coffee on 2000-01-14?
Thought 1: Maybe I can write a SQL query to solve this problem.
Action 1: SQLInterpreter(SELECT Volume FROM coffee.coffee_data WHERE Date = '2000-01-14';
Observation 1: Volume: 10115
Thought 2: The volume of coffee on 2000-01-14 is 10115.
Action 2: Finish[10115]
"""

TOOLQA_HARD3 = """Question: What was the opening price of coffee on 2008-02-07?
Thought 1: The question is asking for the opening price of coffee on a specific date. We need to load the coffee database to find this information.
Action 1: LoadDB[coffee]
Observation 1: We have successfully loaded the coffee database in SQLite, including the following columns: Date, Open, High, Low, Close, Volume, Currency.
Thought 2: We need to filter the database to find the entry for the date 2008-02-07 to get the opening price.
Action 2: FilterDB[Date=2008-02-07]
Observation 2: We have successfully filtered the data (1 rows).
Thought 3: We need to retrieve the opening price from the filtered data for the date 2008-02-07.
Action 3: GetValue[Open]
Observation 3: 142.6
Thought 4: The opening price of coffee on 2008-02-07 is 142.6.
Action 4: Finish[142.6]
Observation 4: Answer is CORRECT
Correct answer: 142.6

Question: What was the lowest coffee price from 2011-11-04 to 2012-03-21?
Thought 1: To answer this question, I need to load the coffee database first.
Action 1: LoadDB[coffee]
Observation 1: We have successfully loaded the coffee database in SQLite, including the following columns: Date, Open, High, Low, Close, Volume, Currency.
Thought 2: To find the lowest coffee price in the specified date range, I need to filter the database for dates from 2011-11-04 to 2012-03-21.
Action 2: SQLInterpreter[SELECT MIN(Low) FROM coffee_data WHERE Date >= '2011-11-04' AND Date <= '2012-03-21';]
Observation 2: MIN(Low): 181.35
Thought 3: The lowest coffee price from 2011-11-04 to 2012-03-21 is 181.35.
Action 3: Finish[181.35]
Observation 3: Answer is CORRECT
Correct answer: 181.35 USD
"""
COTQA_SIMPLE6 = """Question: How many extra minutes did the DL1575 flight take from ATL to MCO on 2022-01-12?
Let's think step by step. I need to first load the flights database with LoadDB[flights]. And then, I need to filter the information I want according to the query with FilterDB[Flight_Number_Marketing_Airline=1575, FlightDate=2022-01-12, Origin=ATL, Dest=MCO]. Then, I need to get the value of the departure time from the database with GetValue[DepDelay]. We then need to know the arrival delayed time with GetValue[ArrDelay]. To compute the extra minutes, we need to subtract the departure delayed time from the arrival delayed time with Calculate[(-17)-(-7)].  After calculation, we know that the answer is -10.
Action 1: LoadDB[flights]-->FilterDB[Flight_Number_Marketing_Airline=1575, FlightDate=2022-01-12, Origin=ATL, Dest=MCO]-->GetValue[DepDelay]-->GetValue[ArrDelay]-->Calculate[(-17)-(-7)]-->Finish[-10]
"""
COT = """
"""

COT_REFLECT = """
"""

REFLECTIONS = """
"""