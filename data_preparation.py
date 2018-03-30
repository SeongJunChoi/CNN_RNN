#####---(Deep Learning에 사용할 data를 불러와 DL에 넣기 전 상태로 가공하기)---#####
##  설명
#   Deep learning에 사용할 data를 여러 형식에 따라 알맞게 불러오고 DL에 넣을 input data와 target(or label) data를 만드는 library

##  업데이트 기록지
#   2018.03.19.월요일 : EMG 4채널 data와 28개의 marker 좌표(x,y,z,az,el,r) data를 불러오기
#                     1개의 csv 파일은 한 피실험자의 한 번의 실험에서 얻은 EMG 4채널 data와 1개의 marker 좌표(x,y,z,az,el,r) data가 들어있음.
#                     열의 이름은 없음
#   2018.03.26.월요일 : Marker 값을 1000개의 구간으로 나눠 label을 만드는 함수와 label을 가지고 다시 값으로 바꾸는(대표값이 되겠지 실제값이 아닌) 함수 완성


#####------------------------------------------------


### 사용할 lib 소환
#   os lib 소환 (system location을 불러오고 지정하기 위해)
import os
#   glob lib 소환 (여러 파일을 한 번에 불러오기 위해)
import glob
#   csv lib 소환 (csv 파일로 불러오기 위해)
import csv
#   numpy lib 소환 (행렬 계산을 위해)
import numpy as np
#   pandas lib 소환 (csv 파일을 불러오기 위해)
import pandas as pd
#   random lib 소환 (data의 순서를 섞기 위해)
import random
#   tqdm에 있는 trange lib 소환 (progress bar를 출력하기 위해)
from tqdm import trange


### Data가 저장된 폴더에서 여러 하위 폴더에 든 data files의 주소를 한 번에 list 형태로 저장하기 위한 함수
def find_files (folder_name, want_file_type) :

    ##  기본 작업 위치 설정
    #   기본 작업 위치

    main_work_location = 'D:\\CSJ\\Tensorflow\\market_price_(EMG)_lstm\\Data\\CSV Data\\' + folder_name + '\\'
    # main_work_location = os.getcwd() + '\\data\\' + folder_name + '\\'
    #print(main_work_location)

    ##  해당 폴더에 있는 모든 csv 파일의 주소를 저장하기 위한 부분
    #   모든 csv 파일의 주소를 저장할 변수 list로 초기화
    files_list = []
    #   os.walk : 지정된 폴더의 하위 폴더 안까지 체크해주는 함수
    #   path ; 하위 폴더명 list (맨 처음엔 빈칸이 들어감) ex) ~~/, ~~/01, ~~/02, ..., ~~/28
    #   files : 해당 path 폴더에 있는 파일명 list ex) ~~/01 => ~~.csv, ~~.csv, ..., ~~.csv
    for (path, dir, files) in os.walk(main_work_location, topdown=True) :
        for filename in files :
            #   ()안(해당 파일)에 경로에서 확장자 부분과 그 외 부분을 나누고 확장자 명을 저장
            #   ex) ~~~\~~.exe를 '~~~\~~'과 '.exe'로 나눔
            now_file_type = os.path.splitext(filename)[-1]
            #   저장한 확장자 명이 입력한 확장자 명과 같을 경우, 해당 파일의 주소를 저장
            if now_file_type == want_file_type :
                #   path.split((os.getcwd()+'\\'))[-1]의 예제 : data\Data_for_LSTM\01\cam&emg_01sub_02comb_01mark_trial01.csv
                now_file_location = path.split((os.getcwd()+'\\'))[-1] + '\\' + filename
                files_list.append(now_file_location)

    ##  모든 csv 파일의 주소가 저장된 변수를 리턴
    ##  files_list = ['~~/~~.csv', '~~/~~.csv', ..., '~~/~~.csv'] (길이 : 8820=315*28)
    return  files_list


### 저장한 files의 위치들을 바탕으로 data 불러오기 위한 함수
def load_data (files_list, want_file_type, data_column_names) :

    ##  불러온 data와 해당 file명을 저장하기 위해 초기화
    #   불러온 data를 딕셔너리 하나에 저장하기 위해 초기화
    all_data = {}
    #   불러온 data의 파일명을 list 형태로 저장하기 위해 초기화
    all_data_name = []


    ##  불러올 files의 확장자가 csv 일때
    if want_file_type == '.csv' :

        #   File 하나씩 들어있는 data를 불러옴.
        for i in trange(len(files_list)) :
            #   해당 file 안에 든 data를 불러옴.
            now_data = pd.read_csv(files_list[i], names=data_column_names, encoding='euc-kr')
            #   불러온 data를 배열 형태로 변환
            now_data_array = np.array(now_data)
            #   data가 들어있던 file의 이름
            now_data_name = files_list[i].split('\\')[-1].split('.')[0]
            #   data가 들어있던 file의 이름을 전체 files 이름을 저장할 list 변수에 추가
            all_data_name.append(now_data_name)
            #   배열로 바뀐 data를 딕셔너리 변수에 file명을 이용하여 저장
            all_data[now_data_name] = now_data_array

            '''
            #   진행중인걸 표시하기 위해
            if (i % 100 == 0) or (i == (len(files_list) - 1)) :
                print('전체 ', len(files_list), '의 files 중에서 ', (i+1), '번째 file을 불러오고 있습니다.')

            #   맨 처음과 맨 마지막 파일의 정보를 출력해서 잘 불러오고 있는지 체크
            if (i == 0) or (i == (len(files_list) - 1)) :
                now_data.info()
            '''

    ##  모든 data가 저장된 딕셔너리 변수와 모든 files의 이름이 담긴 list 변수를 리턴
    return  all_data, all_data_name


### Data set에서 input data와 output의 기준이 되는 target data를 나누기 위한 함수
def separate_data (total_data, total_file_names, EMG_ch_num, marker_axis_type) :

    ##  Input data와 target data를 각각 하나의 딕셔너리로 저장하기 위해 초기화
    #   Input data set
    all_input_data = {}
    #   Target data set
    all_target_data = {}

    ##  주어진 EMG 채널과 marker 축 개수에 따라 input data와 target data 나누기
    for i in trange(len(total_file_names)):

        ##  해당 key 안에 든 파일을 가지고 input data와 target data로 나누기
        #   Input data
        now_input_data = total_data[total_file_names[i]][:, :EMG_ch_num]
        #   Target data
        if marker_axis_type == 3 :
            now_target_data = total_data[total_file_names[i]][:, EMG_ch_num:(EMG_ch_num + marker_axis_type)]
        else :
            now_target_data = total_data[total_file_names[i]][:, (EMG_ch_num + marker_axis_type):(EMG_ch_num + marker_axis_type + 1)]

        ##  나눈 두 개의 data를 각각의 딕셔너리에 저장
        #   Input data
        all_input_data[total_file_names[i]] = now_input_data
        #   Target data
        all_target_data[total_file_names[i]] = now_target_data

    ##  나뉜 input data set과 target data set을 리턴
    return all_input_data, all_target_data


### 원하는 시퀀스 길이에 맞게 data 편집하기 위한 함수
def adjusted_sequence (input_data_dic, target_data_dic, key_name_list, seq_length) :
    ##  원하는 시퀀스 길이로 편집된 input data와 target data를 각각 하나의 딕셔너리로 저장하기 위해 초기화
    #   Input data set
    all_adjusted_input = {}
    #   Target data set
    all_adjusted_target = {}
    #   해당 data들의 길이를 저장
    all_adjusted_length = {}

    ##  원하는 시퀀시 길이에 맞게 data 편집하는 부분
    ##  (len(y) - seq_length)이 되면 시퀀스로 뽑혀지는 구간이 한 개가 적으므로 +1 해주어 뽑히는 구간이 끝까지 맞도록 설정
    for i in trange(len(key_name_list)):

        ##  딕셔너리에 key 별로 넣기 위해 한 key에 해당하는 data들을 차근차근 저장하기 위한 변수를 list로 초기화
        now_input_set = []
        now_target_set = []

        #   해당 key 값의 길이
        now_lenght = len(target_data_dic[key_name_list[i]]) - seq_length + 1
        for j in range(0, now_lenght) :
            ##  해당 key 안에 든 data를 가지고 길이 조정
            #   Input data
            #   붙이기 쉽게 잠시 배열을 list로 변환
            now_input_data = list(input_data_dic[key_name_list[i]][j : (j+seq_length)])
            #   List에 합침
            now_input_set.append(now_input_data)
            #   Target data
            #   붙이기 쉽게 잠시 배열을 list로 변환
            now_target_data = list(target_data_dic[key_name_list[i]][(j+seq_length-1)])
            #   List에 합침
            now_target_set.append(now_target_data)

        ##  딕셔너리에 배열 형태로 저장하기 위해 list를 배열로 변환
        now_input_set = np.array(now_input_set)
        now_target_set = np.array(now_target_set)

        ##  나눈 두 개의 data를 각각의 딕셔너리에 저장
        #   Input data
        all_adjusted_input[key_name_list[i]] = now_input_set
        #   Target data
        all_adjusted_target[key_name_list[i]] = now_target_set
        #   길이
        all_adjusted_length[key_name_list[i]] = now_lenght

    ##  원하는 시퀀스 길이 맞게 편집된 input data set과 target data set을 리턴
    return all_adjusted_input, all_adjusted_target, all_adjusted_length


### Data를 랜덤하게 섞는 효과를 주기 위해 딕셔너리 key명과 같은 file_names list 항목을 랜덤하게 섞음
def shuffle_data (total_file_names, marker_num, subject_num, trial_num, train_option) :
    ##  File 이름을 랜덤하게 섞은 결과를 저장하기 위한 list 변수
    shuffled_file_names = []

    ##  Train data를 추출하는 방법에 따라 섞는 방법도 다르게 섞음
    if train_option == 0 : # 모든 피실험자에서 동일한 수의 data를 train data로 추출하기 위해 섞는 것도 피실험자 당 섞음
        #   File 이름을 랜덤하게 섞는 부분
        for i in trange((marker_num * subject_num)) :
            #   랜덤하게 섞을 구간을 선택
            chosen_file_name = total_file_names[(i * trial_num):((i + 1) * trial_num)]
            #   랜덤하게 섞기
            random.shuffle(chosen_file_name)
            #   랜덤하게 섞은 것을 한 개의 변수에 합치기
            shuffled_file_names = shuffled_file_names + chosen_file_name

    ##  원하는 조건으로 랜덤하게 섞은 file names list를 리턴
    return shuffled_file_names


### 이번 deep learning에서 사용할 원하는 마커와 원하는 피실험자들 data 선택하기 위한 함수
def choose_data (re_file_names, chosen_marker_num, chosen_subject_num, subject_num, trial_num, train_option) :

    ##  Train data를 추출하는 방법에 따라 섞는 방법도 다르게 섞음
    if train_option == 0 : # 모든 피실험자에서 동일한 수의 data를 train data로 추출하기 위해 뽑는 것도 제한적을 뽑아야 함
        #   선택한 구간의 시작점
        start_point = ((chosen_marker_num[1] - 1) * (subject_num * trial_num)) + ((chosen_subject_num[1] - 1) * trial_num)
        #   선택한 구간의 끝점
        end_point = (((chosen_marker_num[1] - 1) + (chosen_marker_num[0] - 1)) * (subject_num * trial_num)) + ((chosen_subject_num[1] - 1 + chosen_subject_num[0]) * trial_num)
        #   이번 deep learning에서 사용할 피설험자들 관련 file names 선택
        selected_file_names = re_file_names[start_point:end_point]

    ##  선택한 data file name을 리턴
    return selected_file_names


### 원하는 train 비율만큼 train set을 만들고 나머지는 test set을 만들기 위해 train 비율만큼 선택된 file name 중 일부를 train set으로 지정 후 나머지는 test set으로 지정하기 위한 함수
def make_train_test_list_set (chosen_file_names, trial_num, train_ratio, chosen_marker_num, chosen_subject_num, train_option) :
    ##  전체 train file names와 test file names를 저장할 list 변수 초기화
    #   Train 용
    train_files = []
    #   Test 용
    test_files = []

    ##  train option에서 모든 피실험자에서 동일한 수의 data를 train data로 추출하는 것을 선택했을 때
    if train_option == 0 : # 모든 피실험자에서 동일한 수의 data를 train data로 추출
        #   입력한 비율만큼을 이용하여 train set을 모든 피실험자에서 균일하게 추출하기 위해 반복 횟수에 train 비율을 곱해 train size를 설정
        train_size = int(trial_num * train_ratio)
        print('각 피실험자당 train data size는 : ', train_size)
        #   반복 횟수에서 train size를 뺀 값을 test size로 설정
        test_size = trial_num - train_size
        print('각 피실험자당 test data size는 : ', test_size)
        print('train_size ', train_size, '+ test_size ', test_size, ' = 한 피실험자당 반복 측정 횟수 ', trial_num)
        print('위의 수식이 성립해면 제대로 구한 것')

        ##  정해진 train size와 test size를 바탕으로 file names를 train set과 test set으로 나누기
        for i in trange(chosen_marker_num[0] * chosen_subject_num[0]) :
            #   한 피실험자 내에서 train용 file names을 추출
            now_train_names = chosen_file_names[(i * trial_num) : ((i * trial_num) + train_size)]
            #   train 전체 file names을 저장할 list에 추가
            train_files = train_files + now_train_names

            #   한 피실험자 내에서 test용 file names을 추출
            now_test_names = chosen_file_names[((i * trial_num) + train_size): ((i + 1) * trial_num)]
            #   test 전체 file names을 저장할 list에 추가
            test_files = test_files + now_test_names

    ##  정해진 비율에 따라 만들어진 train용 file names와 test용 file names를 저장한 list를 리턴
    return train_files, test_files


### 위에서 train용과 test용 file names를 가지고 실제 data와 target도 train용과 test용으로 나누기 위한 함수(행렬의 행결합은 np.r_())
def make_train_test_set (adjusted_input_set, adjusted_target_set, train_file_names, test_file_names) :
    ##  Train file names과 test file names에 해당하는 data와 target을 train용과 test용으로 나눈 후 저장할 변수를 list로 초기화
    #   Train data용
    train_data = []
    #   Train target용
    train_target = []
    #   Test data용
    test_data = []
    #   Test target용
    test_target = []

    ##  Train data와 train target 만드는 부분
    for i in range(len(train_file_names)) :
        #   Train names list에 든 name을 하나씩 불러와 거기에 해당하는 data와 target를 별도로 추출
        #   List로 해주어야 단순히 +로 행결합을 할 수 있어 list화
        now_data = list(adjusted_input_set[train_file_names[i]])
        now_target = list(adjusted_target_set[train_file_names[i]])
        #   List의 행결합
        train_data = train_data + now_data
        train_target = train_target + now_target

    #   RNN 입력을 위해 다시 배열로 변환
    train_data = np.array(train_data)
    train_target = np.array(train_target)

    ##  Test data와 test target 만드는 부분
    for i in range(len(test_file_names)) :
        #   Train names list에 든 name을 하나씩 불러와 거기에 해당하는 data와 target를 별도로 추출
        #   List로 해주어야 단순히 +로 행결합을 할 수 있어 list화
        now_data = list(adjusted_input_set[test_file_names[i]])
        now_target = list(adjusted_target_set[test_file_names[i]])
        #   List의 행결합
        test_data = test_data + now_data
        test_target = test_target + now_target

    #   RNN 입력을 위해 다시 배열로 변환
    test_data = np.array(test_data)
    test_target = np.array(test_target)

    ##  나눠진 train data set, train target set, test data set, test target set을 리턴
    return train_data, train_target, test_data, test_target


### Target dat를 classification을 위해 0과 1로 이루어진 label을 생성하기 위한 함수
def make_num_label(target_data_dic, key_name_list, class_num) :

    ##  Classification을 위해 만든 각 데이터에 대한 label을 딕셔너리에 쭉 저장하기 위해 초기화
    #   Label 저장용 딕셔너리
    all_label_dic = {}
    #   gap 저장용 딕셔너리
    all_gap_dic = {}

    for i in trange(len(key_name_list)):

        now_target_data = target_data_dic[key_name_list[i]]

        ##  해당 key에 있는 배열의 최대값과 최소값 파악
        now_max = 1.0 #now_target_data.max()
        now_min = 0.0 # now_target_data.min()
        #   원하는 class 수대로 최소~최대 사이를 등분
        gap = (now_max - now_min) / class_num

        ##  해당 key에 있는 배열의 row size와 column size를 저장
        row_num = now_target_data.shape[0]
        #col_num = now_target_data.shape[1]

        ##  변수 초기화
        #   Classification에서는 해당 class 열 부분에만 1이고 나머지는 0인 label이 필요하므로 처음에는 모든 원소가 0인 행렬을 생성
        num_label = np.zeros((row_num, class_num))
        #print('초기화된 숫자 label의 크기는 : ', num_label.shape)
        #   각 class마다 몇 개의 data가 들어가는지 기록하기 위한 행렬 변수 초기화
        count_per_label = np.zeros((1, class_num))

        ##  행 하나씩 불러와 크기를 비교하여 label을 만듬.
        for j in range(row_num):

            ##  원하는 class로 최소-최대 구간을 나눴을 때 해당 행의 값이 어느 구간에 속하는지 판단하여 label 부여
            for k in range(class_num):

                ##  원하는 class로 최소-최대 구간을 나눴을 때 해당 행의 값이 어느 구간에 속하는지 판단하여 구간 번째와 동일한 열에 1을 대입
                if (now_target_data[j] >= (k * gap)) and (now_target_data[j] < ((k + 1) * gap)):
                    #   해당하는 열에 1을 대입 (나머지 열은 0)
                    num_label[j, k] = 1
                    #   해당하는 class에 속하는 개수를 하나 증가
                    count_per_label[0, k] = count_per_label[0, k] + 1

            #   1.0보다 큰 값이면 마지막 class에 속하도록 설정
            if (now_target_data[j] >= now_max):
                num_label[j, (class_num - 1)] = 1
                #   해당하는 class에 속하는 개수를 하나 증가
                count_per_label[0, (class_num - 1)] = count_per_label[0, (class_num - 1)] + 1
            #   0.0보다 작은 값이면 첫번재 class에 속하도록 설정
            elif (now_target_data[j] < now_min):
                num_label[j, 0] = 1
                #   해당하는 class에 속하는 개수를 하나 증가
                count_per_label[0, 0] = count_per_label[0, 0] + 1

        ##  각 key에 해당하는 값을 초기화된 딕셔너리에 저장
        #   Label
        all_label_dic[key_name_list[i]] = num_label
        #   Gap
        all_gap_dic[key_name_list[i]] = gap


    print('각 class 별로 들어있는 data의 개수는 : ')
    for m in range(class_num) :
        print(str(m+1), '번째 class에 들어있는 data의 개수는 : ', count_per_label[0, m])


    ##  Target value에 근거하여 만든 label 딕셔너리와 각 class별 value의 gap을 리턴
    return all_label_dic, all_gap_dic


### Class를 다시 숫자로 바꾸기 위한 함수
def return_value (label_data_dic, key_name_list, class_num, all_gap_dic) :

    ##  Classification을 위해 만든 label에 의거하여 다시 대표값으로 바꾼 결과를 딕셔너리에 쭉 저장하기 위해 초기화
    all_label_value_dic = {}

    for i in trange(len(key_name_list)):

        #   차근차근 key에 든 label을 불러옴.
        now_key_array = label_data_dic[key_name_list[i]]
        #   차근차근 key에 든 gap을 불러옴.
        now_gap = all_gap_dic[key_name_list[i]]
        #   현재 key에 들은 data의 row size 저장
        now_key_row_num = now_key_array.shape[0]
        #   저장한 row size를 바탕으로 class별 대표값을 저장할 행렬 초기화
        now_key_value = np.zeros((now_key_row_num, 1))
        #print('초기화된 대표값을 넣을 변수의 크기는 : ', now_key_value.shape)

        ##  행 하나씩 불러와 대표값을 설정
        for j in range(now_key_row_num):

            #   해당 행에서 1이 들어있는 위치를 저장
            one_position = np.where(now_key_array[j] == 1)[0]

            #   위치에 따라 다른 대표값을 저장
            for k in range(class_num):

                if one_position == k:
                    #   해당 행에 1이 있는 위치에 최소~최대 사이를 class의 수로 나눈 gap 만큼을 곱해 대표값으로 설정
                    now_key_value[j] = k * now_gap

        all_label_value_dic[key_name_list[i]] = now_key_value

    ##  Label을 근거로 다시 값(대표값)으로 바꾼 결과 딕셔너리를 리턴
    return all_label_value_dic


