#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import timeit
import time
from datetime import datetime
import os
import sys
import gc

def load_data(path, epsilon):
    start_time = timeit.default_timer()
    path = os.path.expanduser(path)
    fault_code = []
    for (dirname, subdir, subfile) in os.walk(path):
        for f1 in subdir:
            print f1
            if f1 == 'Positive': fault_num = 0
            else: fault_num = 1
            fault_code.append(f1)
            filename = os.path.join(dirname, f1)
            print('\n(' + filename + ')')
            if fault_num == 0:
                dataset = Load_data(filename, fault_num, epsilon)
            else:
                dataset = Add_data(dataset, filename, fault_num, epsilon)
    ins, labels, fault_time, millisecond = dataset
    end_time = timeit.default_timer()
    print ('load dataset ran for %.2fmin' %((end_time - start_time)/60.))
    return ins, labels, fault_code, fault_time, millisecond

def Add_data(data_old, new_path, new_code, epsilon):
    ins_old, outs_old, ft_old, ms_old = data_old
    ins_new, outs_new, ft_new, ms_new = Load_data(path=new_path,
                                                  fault_code=new_code,
                                                  epsilon=epsilon)
    ins_update = ins_old + ins_new
    outs_update = np.hstack((outs_old, outs_new))
    ft_update = ft_old + ft_new
    ms_update = ms_old + ms_new
    return ins_update, outs_update, ft_update, ms_update

def Load_data(path, fault_code, epsilon):
    Weather_index = {}.fromkeys(['gWindSpeed10s', 'gWindSpeed1', 'gWindSpeed2',
                           'gWindDirection1Avg25s', 'gWindDirection2Avg25s'])
    Weather_append = np.zeros((1, 5))
    '''
    天气0：10秒钟平均风速；风速计1风速测量值；风速2风速测量值;
          风向标1风向测量值25秒平均值; 风向标2风向测量值25秒平均值
    '''
    Environment_index = {}.fromkeys(['gTmpNac', 'gTmpNacOutdoor', 'gTmpCabTb',
                                     'gTmpTb', 'gTmpCabNac'])
    Environment_append = np.zeros((1, 5))
    '''
    环境1：风机机舱温度；风机舱位温度；主控塔底柜柜内温度；
         风机塔底温度；风机机舱主控柜温度
    '''
    ControlVariable_index = {}.fromkeys(['gPitSysTagPosSet', 'gYawCode',
                                         'gSCACosPhiSet', 'gScadaPowReduct',
                                         'gBrakeCode', 'gMainLoopNumber'])
    ControlVariable_append = np.zeros((1, 6))
    '''
    控制因素2：设定桨距角位置；风机偏航程序等级；
              系统设定功率因数；系统设定功率工作点；
              风机制动等级；风机运行状态
    '''
    Power_index = {}.fromkeys(['gGridU1', 'gGridU2', 'gGridU3',
                         'gGridIL1', 'gGridIL2', 'gGridIL3',
                         'gGridP', 'gGridQ', 'gGridCosPhi', 'gGridFreq'])
    Power_append = np.zeros((1, 10))
    '''
    电网3：电网A相电压；电网B相电压；电网C相电压；
         电网A相电流；电网B相电流；电网C相电流；
         有功功率；无功功率；功率因数；电网频率
    '''
    PitchSystem_index = {}.fromkeys(['gRotSpeed1', 'gRotSpeed2', 'gRotSpeedSSI',
                               'gPitchPosition',
                               'gPitPosEnc1[0]', 'gPitPosEnc1[1]', 'gPitPosEnc1[2]',
                               'gPitPosEnc2[0]', 'gPitPosEnc2[1]', 'gPitPosEnc2[2]',
                   'gPitAxisBattVolt[0]', 'gPitAxisBattVolt[1]', 'gPitAxisBattVolt[2]',
                      'gbufBattBoxTemp[0]', 'gbufBattBoxTemp[1]', 'gbufBattBoxTemp[2]',
                       'gbufAxisBoxTemp[0]', 'gbufAxisBoxTemp[1]','gbufAxisBoxTemp[2]',
                            'gbufMotorTemp[0]', 'gbufMotorTemp[1]', 'gbufMotorTemp[2]',
                        'gbufHubTemp', 'gPitSpeed[0]', 'gPitSpeed[1]', 'gPitSpeed[2]'])
    PitchSystem_append = np.zeros((1, 26))
    '''
    变桨系统4：模拟量风轮转速1；模拟量风轮转速2；风轮转速；
             桨距角实际位置；
             桨叶1电机编码器角度；桨叶2电机编码器角度；桨叶3电机编码器角度；
             桨叶1冗余编码器角度；桨叶2冗余编码器角度；桨叶3冗余编码器角度；
             桨叶1电池电压；桨叶2电池电压；桨叶3电池电压；
             变桨电池柜1温度；变桨电池柜2温度；变桨电池柜3温度；
             变桨轴柜1温度；变桨轴柜2温度；变桨轴柜3温度；
             变桨电机1温度；变桨电机2温度；变桨电机3温度；
             轮毂温度；变桨1速度；变桨2速度；变桨3速度
    '''
    EgineRoom_index = {}.fromkeys(['gNacVibrationX', 'gNacVibrationY',
                                   'gVibEffectiveValue'])
    EgineRoom_append = np.zeros((1, 3))
    '''
    机舱5：机舱X周方向振动；机舱Y轴方向振动；
           机舱振动有效值
    '''
    GearBox_index = {}.fromkeys(['gTmpGbxHigh', 'gTmpGbxLow', 'gTmpGbxInletOil',
                           'gTmpGbxOil', 'gTmpGbxWater'])
    GearBox_append = np.zeros((1, 5))
    '''
    齿轮箱6：齿轮箱高速轴温度；齿轮箱低速轴温度；齿轮箱进油口油温；
           齿轮箱油温；齿轮箱冷却水温度
    '''
    Generator_index = {}.fromkeys(['gActualGenSpd', 'gTmpGenWinding',
                             'gTmpGenWindingU1', 'gTmpGenWindingV1', 'gTmpGenWindingW1',
                             'gTmpGenBearingF','gTmpGenBearingR', 'gTmpGenRing'])
    Generator_append = np.zeros((1, 8))
    '''
    发电机7：发电机转速；发电机绕组温度；
           发电机转子U1温度；发电机转子V1温度；发电机转子W1温度；
           发电机前轴承温度；发电机后轴承温度；发电机滑环温度；
    '''
    YawSystem_index = {}.fromkeys(['gYawDev25s', 'gYawTwistPosition', 'gHydSysPressure'])
    YawSystem_append = np.zeros((1, 3))
    '''
    偏航系统8：偏航角度；扭缆位置；液压系统压力
    '''
    Converter_index = {}.fromkeys(['gConvTorqueActualVal', 'gTmpConverterCoolWtr'])
    Converter_append = np.zeros((1, 2))
    '''
    变流器9：变流器实际输出转矩百分比；变流器冷却水温度
    '''
    path = os.path.expanduser(path)                         #此处要变成交互数据
    insset_list = []
    outsset = np.array([])
    fault_time = []
    msset = []
    for (dirname, subdir, subfile) in os.walk(path):
        print('[' + dirname + ']')
        for f2 in subfile:
            filename = os.path.join(dirname, f2)
            print(filename)
            index = {}
            Weather = np.zeros((1, 5))
            Environment = np.zeros((1, 5))
            ControlVariable = np.zeros((1, 6))
            Power = np.zeros((1, 10))
            PitchSystem = np.zeros((1, 26))
            EgineRoom = np.zeros((1, 3))
            GearBox = np.zeros((1, 5))
            Generator = np.zeros((1, 8))
            YawSystem = np.zeros((1, 3))
            Converter = np.zeros((1, 2))
            with open(filename, 'r') as f:
                f.readline()
                value_name = f.readline()
                value_name = value_name.strip()
                value_name = value_name.split(',')
                TimeStamp_0 = str('2016_1_1_00:00:00:00')
                TimeStamp_1 = str('2016_1_2_00:00:00:00')
                ms_list = []
                for i in xrange(len(value_name)):
                    index[value_name[i]] = i                  #将原始数据分组标号
                for (k, v) in index.items():
                    if Weather_index.has_key(k) == True:
                        Weather_index[k] = v
                    if Environment_index.has_key(k) == True:
                        Environment_index[k] = v
                    if ControlVariable_index.has_key(k) == True:
                        ControlVariable_index[k] = v
                    if Power_index.has_key(k) == True:
                        Power_index[k] = v
                    if PitchSystem_index.has_key(k) == True:
                        PitchSystem_index[k] = v
                    if EgineRoom_index.has_key(k) == True:
                        EgineRoom_index[k] = v
                    if GearBox_index.has_key(k) == True:
                        GearBox_index[k] = v
                    if Generator_index.has_key(k) == True:
                        Generator_index[k] = v
                    if YawSystem_index.has_key(k) == True:
                        YawSystem_index[k] = v
                    if Converter_index.has_key(k) == True:
                        Converter_index[k] = v
                line_index = 0
                line_index2 = 0
                line2 = f.readlines()
                for line in line2[: -1]:
                    line = line.strip()
                    line = line.split(',')
                    line_index2 += 1
                    TimeStamp_1 = line[0].strip('\x00')
                    if TimeStamp_1 == TimeStamp_0:
                        #ft, ms = ts_ms(TimeStamp_1)
                        ft = ts_ms(TimeStamp_1)
                    if TimeStamp_1 != TimeStamp_0:
                        ms = ts_ms(TimeStamp_1)
                        j_0 = 0
                        for (k, v) in Weather_index.items():
                            if v != None:
                                Weather[line_index, j_0] = line[v]
                            j_0 += 1
                        j_1 = 0
                        for (k, v) in Environment_index.items():
                            if v != None:
                                Environment[line_index, j_1] = line[v]
                            j_1 += 1
                        j_2 = 0
                        for (k, v) in ControlVariable_index.items():
                            if v != None:
                                ControlVariable[line_index, j_2] = line[v]
                            j_2 += 1
                        j_3 = 0
                        for (k, v) in Power_index.items():
                            if v != None:
                                Power[line_index, j_3] = line[v]
                            j_3 += 1
                        j_4 = 0
                        for (k, v) in PitchSystem_index.items():
                            if v != None:
                                PitchSystem[line_index, j_4] = line[v]
                            j_4 += 1
                        j_5= 0
                        for (k, v) in EgineRoom_index.items():
                            if v != None:
                                EgineRoom[line_index, j_5] = line[v]
                            j_5 += 1
                        j_6 = 0
                        for (k, v) in GearBox_index.items():
                            if v != None:
                                GearBox[line_index, j_6] = line[v]
                            j_6 += 1
                        j_7 = 0
                        for (k, v) in Generator_index.items():
                            if v != None:
                                Generator[line_index, j_7] = line[v]
                            j_7 += 1
                        j_8 = 0
                        for (k, v) in YawSystem_index.items():
                            if v != None:
                                YawSystem[line_index, j_8] = line[v]
                            j_8 += 1
                        j_9 = 0
                        for (k, v) in Converter_index.items():
                            if v != None:
                                Converter[line_index, j_9] = line[v]
                            j_9 += 1

                        line_index += 1
                        TimeStamp_0 = TimeStamp_1
                        Weather = np.vstack((Weather, Weather_append))
                        Environment = np.vstack((Environment, Environment_append))
                        ControlVariable = np.vstack((ControlVariable, ControlVariable_append))
                        Power = np.vstack((Power, Power_append))
                        PitchSystem = np.vstack((PitchSystem, PitchSystem_append))
                        EgineRoom = np.vstack((EgineRoom, EgineRoom_append))
                        GearBox = np.vstack((GearBox, GearBox_append))
                        Generator = np.vstack((Generator, Generator_append))
                        YawSystem = np.vstack((YawSystem, YawSystem_append))
                        Converter = np.vstack((Converter, Converter_append))
                        ms_list.append(ms)
            Weather = np.delete(Weather, -1, axis = 0)
            Environment = np.delete(Environment, -1, axis = 0)
            ControlVariable = np.delete(ControlVariable, -1, axis = 0)
            Power = np.delete(Power, -1, axis = 0)
            PitchSystem = np.delete(PitchSystem, -1, axis = 0)
            EgineRoom = np.delete(EgineRoom, -1, axis = 0)
            GearBox = np.delete(GearBox, -1, axis = 0)
            Generator = np.delete(Generator, -1, axis = 0)
            YawSystem = np.delete(YawSystem, -1, axis = 0)
            Converter = np.delete(Converter, -1, axis = 0)

            fault_level2 = np.array(fault_code)
            insset = np.hstack((Weather, Environment, ControlVariable, Power, PitchSystem,
                      EgineRoom, GearBox, Generator, YawSystem, Converter))
            #insset_new = data_preprocessing(insset, epsilon)
            #insset_new = data_preprocessing3(insset)
            #insset_new = SetData(insset)
            outsset = np.hstack((outsset,fault_level2)).astype('int32')
            insset_list.append(insset)
            fault_time.append(ft)
            msset.append(ms_list)
    return insset_list, outsset, fault_time, msset

def data_seg(x, v_name):
    data_name = dict()
    data_name['Weather'] = 0
    data_name['Environment'] = 1
    data_name['ControlVariable'] = 2
    data_name['Power'] = 3
    data_name['PitchSystem'] = 4
    data_name['EngineRoom'] = 5
    data_name['GearBox'] = 6
    data_name['Generator'] = 7
    data_name['YawSystem'] = 8
    data_name['Converter'] = 9
    data = x[:, data_name[v_name]]
    return data

def dataset_trans(dataset):
    x = dataset
    x0 = x[0]
    for i in range(len(x)):
        if  (i != 0):
            x1 = x[i]
            x0 = np.vstack((x0, x1))
        gc.collect()
    return x0

def samples_nor(x, limits, k):
    minmaxfun = dict()
    minmaxfun[0] = minmax_standardization4
    minmaxfun[1] = minmax_standardization5
    x_nor = minmaxfun[k](x, limits[0], limits[1])
    return x_nor

def samples_tnor(x, limits, k):
    xlist_nor = []
    for i in range(len(x)):
        x_nor = samples_nor(x[i], limits, k)
        xlist_nor.append(x_nor)
    return xlist_nor

def samples_tnor2(x, limits, k):
    #x_trans = dataset_trans(x)
    x_nor = samples_nor(x, limits, k)
    gc.collect()
    return x_nor

def data_conb(x1, x2):
    x = []
    for i in range(len(x1)):
        x.append(np.hstack((x1[i], x2[i])))
    return x

def minmax_standardization4(x, x_min, x_max):
    for j in range(x.shape[1]):
        if x_min[j] < 0:
            x = np.hstack((x, np.zeros((len(x), 1))))
            for i in range(x.shape[0]):
                if x[i, j] <= x_min[j]:
                    x[i, j] = 1.
                    x[i, -1] = 1.
                elif x[i, j] > x_min[j] and x[i, j] < 0:
                    x[i, j] = 1 * x[i, j] / x_min[j]
                    x[i, -1] = 1.
                elif x[i, j] >= 0 and x[i, j] < x_max[j]:
                    x[i, j] = x[i, j] / x_max[j]
                else: x[i, j] = 1.
        else:
            for i in range(x.shape[0]):
                if x[i, j] < x_min[j]:
                    x[i, j] = 0.
                elif x[i, j] > x_max[j]:
                    x[i, j] = 1.
                else:
                    x[i, j] = (x[i, j] - x_min[j]) / (x_max[j] - x_min[j])
    return x

def minmax_standardization5(x, x_min, x_max):
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if x[i, j] <= x_min[j]:
                x[i, j] = 0.
            elif x[i, j] >= x_max[j]:
                x[i, j] = 1.
            else: x[i, j] = (x[i, j] - x_min[j]) / (x_max[j] - x_min[j])
    return x

def fea_standardization(x, x_mean, x_std):
    '''
    this function realizes data feature standardization.
    The data is converted to a mean of 0 and variance data 1.
    '''
    x = x - x_mean
    x = x / x_std
    return x

def fea_standarverse(x, x_mean, x_std):
    x *= x_std
    x += x_mean
    return x

def get_usv(x, x_mean, x_std):
    x = fea_standardization(x, x_mean, x_std)
    cov = np.dot(x.T, x) / x.shape[0]
    u, s, v = np.linalg.svd(cov)
    return u, s

def zca_whitening(x, x_mean, x_std, x_u, x_s, epsilon):
    '''
    this function is aimed to reduce the relevance of data and noises.
    '''
    x = fea_standardization(x, x_mean, x_std)
    xrot = np.dot(x, x_u)
    xpcawhite = xrot / np.sqrt(x_s + epsilon)
    xzcawhite = np.dot(xpcawhite, x_u.T)
    xzcawhite = fea_standarverse(x, x_mean, x_std)
    return xzcawhite

def ts_ms(ts):
    fault_timestamp = str(ts)
    fault_timestamp_1 = datetime.strptime(fault_timestamp,'%Y_%m_%d_%H:%M:%S:%f')
    fault_timestamp_2 = fault_timestamp_1.strftime('%Y-%m-%d %H:%M:%S:%f')
    millisecond =  int(time.mktime(fault_timestamp_1.timetuple()))
    return fault_timestamp_2, millisecond

def weather_pre(x):
    wea_limits = np.transpose(pd.read_csv('weather_limits.csv').values[:, 3:5])
    #weathernor = samples_tnor(x, wea_limits, k=0)
    weathernor = samples_tnor2(x, wea_limits, k=0)
    return weathernor

def environment_pre(x):
    env_limits = np.transpose(pd.read_csv('environment_limits.csv').values[:, 3:5])
    environmentnor = samples_tnor2(x, env_limits, k=1)
    return environmentnor

def controller_pre(x):
    control_limits = np.transpose(pd.read_csv('controlvariable_limits.csv').values[:, 3:5])
    controlvariablenor = samples_tnor2(x, control_limits, 0)
    return controlvariablenor

def power_pre(x):
    power_limits = np.transpose(pd.read_csv('power_limits.csv').values[:, 3:5])
    powernor = samples_tnor2(x, power_limits, k=0)
    return powernor

def pitchsystem_pre(x):
    pit_limits = np.transpose(pd.read_csv('pitchsystem_limits.csv').values[:, 3:5])
    #pitch1_limits = np.hstack((pit_limits[:, 5:6], pit_limits[:, 8:9],
    #                           pit_limits[:, 13:14], pit_limits[:, 15:17],
    #                           pit_limits[:, 19:21], pit_limits[:, 22:25]))
    pitch2_limits = np.hstack((pit_limits[:, 0:5], pit_limits[:, 6:8],
                               pit_limits[:, 9:13], pit_limits[:, 14:15],
                               pit_limits[:, 17:19], pit_limits[:, 21:22],
                               pit_limits[:, 25:26]))
    #x = dataset_trans(x)
    #x = np.vstack((x))
    #np.save('../data/pitchsystem_datatransraw', x)
    #print 'pitch1 sel'
    #pitch1 = np.hstack((x[:, 5:6], x[:, 8:9], x[:, 13:14],
    #                    x[:, 15:17], x[:, 19:21], x[:, 22:25]))
    #x = np.delete(x, [5, 8, 13, 15, 16, 19, 20, 22, 23, 24], axis=1)
    #print 'pitch2 sel'
    #pitch2 = np.hstack((x[:, 0:5], x[:, 6:8], x[:, 9:13],
    #                    x[:, 14:15], x[:, 17:19], x[:, 21:22],
    #                   x[:, 25:26]))
    #np.save('../data2/pitch2', pitch2)
    #del x
    #gc.collect()
    #pitch1 = []
    #pitch2 = []
    #for i in range(len(x)):
    #    pitch1.append(np.hstack((x[i][:, 5:6], x[i][:, 8:9], x[i][:, 13:14],
    #                             x[i][:, 15:17], x[i][:, 19:21], x[i][:, 22:25])))
    #    pitch2.append(np.hstack((x[i][:, 0:5], x[i][:, 6:8], x[i][:, 9:13],
    #                             x[i][:, 14:15], x[i][:, 17:19], x[i][:, 21:22],
    #                             x[i][:, 25:26])))
    #print 'pitch1 nor'
    #pitch1 = samples_tnor2(pitch1, pitch1_limits, k=1)
    print 'pitch2 nor'
    x = samples_tnor2(x, pitch2_limits, k=0)

    #print 'conb'
    #pitchnor1 = samples_nor(pitch1, pitch1_limits, k=1)
    #x = samples_nor(x, pitch2_limits, k=0)
    #pitch1 = np.hstack((pitch1, pitch2))
    #pitchsystemnor = data_conb(pitchnor1, pitchnor2)
    #return pitchsystemnor
    #del pitch2
    #gc.collect()
    return x

def engineroom_pre(x):
    eng_limits = np.transpose(pd.read_csv('engineroom_limits.csv').values[:, 3:5])
    engineroomnor = samples_tnor2(x, eng_limits, k=0)
    return engineroomnor

def gearbox_pre(x):
    gea_limits = np.transpose(pd.read_csv('gearbox_limits.csv').values[:, 3:5])
    gearboxnor = samples_tnor2(x, gea_limits, k=1)
    return gearboxnor

def generator_pre(x):
    gen_limits = np.transpose(pd.read_csv('generator_limits.csv').values[:, 3:5])
    gen1_limits = np.delete(gen_limits, 1, axis=1)
    gen2_limits = gen1_limits[:, 1:2]
    '''
    gen1 =[]
    gen2 = []
    for i in range(len(x)):
        gen1.append(np.delete(x[i], 1, axis=1))
        gen2.append(x[i][:, 1:2])
    '''
    gen1 = np.delete(x, 1, axis=1)
    gen2 = x[:, 1:2]
    gennor1 = samples_tnor2(gen1, gen1_limits, k=1)
    gennor2 = samples_tnor2(gen2, gen2_limits, k=0)
    generatornor = np.hstack((gennor1, gennor2))
    #generatornor = data_conb(gennor1, gennor2)
    return generatornor

def yawsystem_pre(x):
    yaw_limits = np.transpose(pd.read_csv('yawsystem_limits.csv').values[:, 3:5])
    yaw1_limits = yaw_limits[:, 0:1]
    yaw2_limits = np.delete(yaw_limits, 0, axis=1)
    '''
    yaw1 =[]
    yaw2 = []
    for i in range(len(x)):
        yaw1.append(x[i][:, 0:1])
        yaw2.append(np.delete(x[i], 0, axis=1))
    '''
    yaw1 = x[:, 0:1]
    yaw2 = np.delete(x, 0, axis=1)
    yawnor1 = samples_tnor2(yaw1, yaw1_limits, k=1)
    yawnor2 = samples_tnor2(yaw2, yaw2_limits, k=0)
    yawsystemnor = np.hstack((yawnor1, yawnor2))
    #yawsystemnor = data_conb(yawnor1, yawnor2)
    return yawsystemnor

def converter_pre(x):
    conv_limits = np.transpose(pd.read_csv('converter_limits.csv').values[:, 3:5])
    converternor = samples_tnor2(x, conv_limits, k=1)
    return converternor

def predata1():
    path = '../data/rawdata'
    x_train, y_train, fault_code, ft_train, ms_train = load_data(path, 0.01)
    np.save('../data/expdata/x_train', x_train)
    np.save('../data/expdata/y_train', y_train)
    np.save('../data/expdata/ft_train', ft_train)
    np.save('../data/expdata/ms_train', ms_train)
    #np.save('../data/unlabeled_faultcode_train', fault_code)
    #print x_test.shape
    #print len(x_test), len(y_test)
    #print fault_code
    '''
    path1 = '../data/labeled_x_train.npy'
    path2 = '../data/unlabeled_x_train50.npy'
    path3 = '../data/unlabeled_x_train100.npy'
    path4 = '../data/unlabeled_x_train150.npy'
    path5 = '../data/unlabeled_x_train200.npy'
    lxt_data = np.load(path1)
    print lxt_data.shape
    uxt50_data = np.load(path2)
    print uxt50_data.shape
    uxt100_data = np.load(path3)
    print uxt100_data.shape
    uxt150_data = np.load(path4)
    print uxt150_data.shape
    uxt200_data = np.load(path5)
    print uxt200_data.shape
    axt_data = np.vstack((lxt_data, uxt50_data, uxt100_data, uxt150_data, uxt200_data))
    wraw = data_seg(axt_data, 'Weather')
    np.save('../data/weather_data', wraw)
    enmraw = data_seg(axt_data, 'Environment')
    np.save('../data/environment_data', enmraw)
    corraw = data_seg(axt_data, 'ControlVariable')
    np.save('../data/controller_data', corraw)
    praw = data_seg(axt_data, 'Power')
    np.save('../data/power_data', praw)
    pitraw = data_seg(axt_data, 'PitchSystem')
    np.save('../data/pitchsystem_data', pitraw)
    engraw = data_seg(axt_data, 'EngineRoom')
    np.save('../data/engineroom_data', engraw)
    gearaw = data_seg(axt_data, 'GearBox')
    np.save('../data/gearbox_data', gearaw)
    genraw = data_seg(axt_data, 'Generator')
    np.save('../data/generator_data', genraw)
    yawraw = data_seg(axt_data, 'YawSystem')
    np.save('../data/yawsystem_data', yawraw)
    covraw = data_seg(axt_data, 'Converter')
    np.save('../data/converter_data', covraw)
    '''

def predata2():
    path = '../data/power_data.npy'
    datax = np.load(path)
    dataxtrans = dataset_trans(datax)

def predata3():
    print 'load data'
    xraw = np.array(np.load('../data/pitchsystem_data.npy'))
    print 'normalize data'
    xnor = pitchsystem_pre(xraw)
    del xraw
    gc.collect()
    np.save('pitch2_datatransnor', xnor)
    #xnor = np.load('../data/yawsystem_datatransnor.npy')
    #np.savetxt('pitchsystem_mean.csv', np.mean(xnor, axis=0), delimiter=',')
    #np.savetxt('pitchsystem_std.csv', np.std(xnor, axis=0), delimiter=',')
    #print 'get u and s'
    #print np.mean(xnor, axis=0), np.std(xnor, axis=0)
    #u, s = get_usv(xnor[:, 0:26], np.mean(xnor, axis=0)[0:26], np.std(xnor, axis=0)[0:26])
    #print 'save parameters'
    #np.savetxt('pitchsystem_u.csv', u, delimiter=',')
    #np.savetxt('pitchsystem_s.csv', s, delimiter=',')
    #print xnor.shape
    #print 'zca white data'
    #x_mean = np.loadtxt('pitchsystem_mean.csv', delimiter=',')[0:26]
    #x_std = np.loadtxt('pitchsystem_std.csv', delimiter=',')[0:26]
    #x_u = np.loadtxt('pitchsystem_u.csv', delimiter=',')
    #x_s = np.loadtxt('pitchsystem_s.csv', delimiter=',')
    #xzca = zca_whitening(xnor[:, 0:26], x_mean=x_mean, x_std=x_std, x_u=x_u, x_s=x_s,
    #                     epsilon=0.01)

def main():
    start_time = timeit.default_timer()
    predata1()
    #predata3()
    end_time = timeit.default_timer()
    print 'the process run for %.2fmin' %((end_time-start_time)/60.)

if __name__ == '__main__':
    main()
