#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import sys
import timeit

import numpy as np
from functions import *

def data_preprocessing3(insset):
    Weather = insset[0]
    PitchSystem = insset[1]
    EgineRoom = insset[2]
    ControlVariable = insset[3]
    GearBox = insset[4]
    Generator = insset[5]
    YawSystem = insset[6]
    Power = insset[7]
    Converter = insset[8]
    FanBrake = insset[9]

    WeatherLimits = np.array([[0., -180., 0., 0., -180.],
                                 [32., 180., 32., 32., 180.]]) #天气变量上下限
    PitchSystemLimits = np.zeros((2, 27)) #变桨系统变量上下限
    PitchSystemLimits[:, 0: 10] = np.array([[0., 200., 0., 0., -1., -10., 200., 0., 0., 0.],
                                               [96., 460., 19., 19., 5., 65., 460., 96., 90., 96.]])
    PitchSystemLimits[:, 10: 20] = np.array([[-1., 0., 0., 0., -10., 0., -10., -60., 0., 200.],
                                                [5., 91., 96., 96., 60., 91., 65., 140., 19., 460.]])
    PitchSystemLimits[:, 20: 27] = np.array([[0., 0., 0., -60., -60., -10., -1.],
                                                [90., 90., 96., 140., 140., 65., 5.]])
    EgineRoomLimits = np.array([[0., 0., 0.],
                                   [0.5, 5., 0.5]]) #机舱振动变量上下限
    ControlVariableLimits = np.array([[-20., -5., -20., -20., -5.],
                                         [55., 55., 55., 20., 55.]]) #控制因素变量上下限
    GearBoxLimits = np.array([[0., -5., 0., 0., 0.],
                                 [90., 68., 80., 90., 80.]]) #齿轮箱变量上下限
    GeneratorLimits = np.array([[0., 0., 0., 0., 0., 0., 0., 0.],
                                   [95., 2300., 145., 145., 95., 145., 145., 90.]])
                                   #发电机变量上下限
    YawSystemLimits = np.array([[0., 0., -750., -180.],
                                   [170., 120., 750., 180.]]) #偏航系统变量上下限
    PowerLimits = np.array([[0., 0., 0., 340., 340., 340., -1., 48.5, -1., 0., -825., -10.],
                               [2000., 2000., 2000., 440., 440., 440., 1., 51.5, 1., 1., 1015., 2350.]])
                                #电网变量上下限
    ConverterLimits = np.array([[-5., 0.],
                                   [70., 100.]]) #变流器变量上下限
    FanBrakeLimits = np.array([[0., 1.],
                                  [70., 9.]]) #风机制动变量上下限
    # data minmax standardization
    WeatherNor = minmax_standardization(Weather, WeatherLimits[0, :], WeatherLimits[1, :])
    PitchSystemNor = minmax_standardization(PitchSystem, PitchSystemLimits[0, :], PitchSystemLimits[1, :])
    EgineRoomNor = minmax_standardization(EgineRoom, EgineRoomLimits[0, :], EgineRoomLimits[1, :])
    ControlVariableNor = minmax_standardization(ControlVariable, ControlVariableLimits[0, :], ControlVariableLimits[1, :])
    GearBoxNor = minmax_standardization(GearBox, GearBoxLimits[0, :], GearBoxLimits[1, :])
    GeneratorNor = minmax_standardization(Generator, GeneratorLimits[0, :], GeneratorLimits[1, :])
    YawSystemNor = minmax_standardization(YawSystem, YawSystemLimits[0, :], YawSystemLimits[1, :])
    PowerNor = minmax_standardization(Power, PowerLimits[0, :], PowerLimits[1, :])
    ConverterNor = minmax_standardization(Converter, ConverterLimits[0, :], ConverterLimits[1, :])
    FanBrakeNor = minmax_standardization(FanBrake, FanBrakeLimits[0, :], FanBrakeLimits[1, :])
    datasetnor_ins = [WeatherNor, PitchSystemNor, EgineRoomNor,ControlVariableNor,
                      GearBoxNor, GeneratorNor, YawSystemNor, PowerNor, ConverterNor,
                      FanBrakeNor]
    datasetnor_ins2 = SetData(datasetnor_ins)
    return datasetnor_ins2

def data_preprocessing(insset, epsilon):
    Weather = insset[0]
    PitchSystem = insset[1]
    EgineRoom = insset[2]
    ControlVariable = insset[3]
    GearBox = insset[4]
    Generator = insset[5]
    YawSystem = insset[6]
    Power = insset[7]
    Converter = insset[8]
    FanBrake = insset[9]

    WeatherLimits = np.array([[0., -180., 0., 0., -180.],
                                 [32., 180., 32., 32., 180.]]) #天气变量上下限
    PitchSystemLimits = np.zeros((2, 27)) #变桨系统变量上下限
    PitchSystemLimits[:, 0: 10] = np.array([[0., 200., 0., 0., -1., -10., 200., 0., 0., 0.],
                                               [96., 460., 19., 19., 5., 65., 460., 96., 90., 96.]])
    PitchSystemLimits[:, 10: 20] = np.array([[-1., 0., 0., 0., -10., 0., -10., -60., 0., 200.],
                                                [5., 91., 96., 96., 60., 91., 65., 140., 19., 460.]])
    PitchSystemLimits[:, 20: 27] = np.array([[0., 0., 0., -60., -60., -10., -1.],
                                                [90., 90., 96., 140., 140., 65., 5.]])
    EgineRoomLimits = np.array([[0., 0., 0.],
                                   [0.5, 5., 0.5]]) #机舱振动变量上下限
    ControlVariableLimits = np.array([[-20., -5., -20., -20., -5.],
                                         [55., 55., 55., 20., 55.]]) #控制因素变量上下限
    GearBoxLimits = np.array([[0., -5., 0., 0., 0.],
                                 [90., 68., 80., 90., 80.]]) #齿轮箱变量上下限
    GeneratorLimits = np.array([[0., 0., 0., 0., 0., 0., 0., 0.],
                                   [95., 2300., 145., 145., 95., 145., 145., 90.]])
                                   #发电机变量上下限
    YawSystemLimits = np.array([[0., 0., -750., -180.],
                                   [170., 120., 750., 180.]]) #偏航系统变量上下限
    PowerLimits = np.array([[0., 0., 0., 340., 340., 340., -1., 48.5, -1., 0., -825., -10.],
                               [2000., 2000., 2000., 440., 440., 440., 1., 51.5, 1., 1., 1015., 2350.]])
                                #电网变量上下限
    ConverterLimits = np.array([[-5., 0.],
                                   [70., 100.]]) #变流器变量上下限
    FanBrakeLimits = np.array([[0., 1.],
                                  [70., 9.]]) #风机制动变量上下限
    # data minmax standardization
    WeatherNor = minmax_standardization(Weather, WeatherLimits[0, :], WeatherLimits[1, :])
    PitchSystemNor = minmax_standardization(PitchSystem, PitchSystemLimits[0, :], PitchSystemLimits[1, :])
    EgineRoomNor = minmax_standardization(EgineRoom, EgineRoomLimits[0, :], EgineRoomLimits[1, :])
    ControlVariableNor = minmax_standardization(ControlVariable, ControlVariableLimits[0, :], ControlVariableLimits[1, :])
    GearBoxNor = minmax_standardization(GearBox, GearBoxLimits[0, :], GearBoxLimits[1, :])
    GeneratorNor = minmax_standardization(Generator, GeneratorLimits[0, :], GeneratorLimits[1, :])
    YawSystemNor = minmax_standardization(YawSystem, YawSystemLimits[0, :], YawSystemLimits[1, :])
    PowerNor = minmax_standardization(Power, PowerLimits[0, :], PowerLimits[1, :])
    ConverterNor = minmax_standardization(Converter, ConverterLimits[0, :], ConverterLimits[1, :])
    FanBrakeNor = minmax_standardization(FanBrake, FanBrakeLimits[0, :], FanBrakeLimits[1, :])
    datasetnor_ins = [WeatherNor, PitchSystemNor, EgineRoomNor,ControlVariableNor,
                      GearBoxNor, GeneratorNor, YawSystemNor, PowerNor, ConverterNor,
                      FanBrakeNor]
    datasetnor_ins2 = SetData(datasetnor_ins)
    # data zca whitening
    datasetzca_ins = zca_whitening(datasetnor_ins2, epsilon)
    return datasetzca_ins

def data_preprocessing2(insset, epsilon):
    dataset_ins = insset
    dataset_ins2 = SetData(dataset_ins)
    datasetnor_ins = fea_standardization(dataset_ins2)
    datasetzca_ins = zca_whitening(datasetnor_ins, epsilon)
    return datasetzca_ins

def get_limits():
    WeatherLimits = np.array([[0., -216., 0., 0., -216.],
                                 [38.4, 216., 38.4, 38.4, 216.]]) #天气变量上下限
    PitchSystemLimits = np.zeros((2, 27)) #变桨系统变量上下限
    PitchSystemLimits[:, 0: 10] = np.array([[-15.1, 0., -1.9, -1.9, -12., -51., 0., -15.1, -51, -15.1],
                                               [106.1, 552., 20.9, 20.9, 12., 81., 552., 106.1, 81., 106.1]])
    PitchSystemLimits[:, 10: 20] = np.array([[-12., 0., -15.1, -15.1, -51., -25.2, -51., -75., -1.9, 0.],
                                                [12., 91., 106.1, 106.1, 81., 116.2, 81., 225., 20.9, 552.]])
    PitchSystemLimits[:, 20: 27] = np.array([[-51., -51., -15.1, -75., -75., -51., -12.],
                                                [81., 81., 106.1, 225., 225., 81., 12.]])
    EgineRoomLimits = np.array([[-0.6, -0.6, -0.1],
                                   [0.6, 0.6, 0.6]]) #机舱振动变量上下限
    ControlVariableLimits = np.array([[-63., -63., -63., -63., -63.],
                                         [93., 93., 93., 93., 93.]]) #控制因素变量上下限
    GearBoxLimits = np.array([[-23., -10., -10, -23., -10.],
                                 [133., 110., 110., 133., 110.]]) #齿轮箱变量上下限
    GeneratorLimits = np.array([[-23., -250., -14.5, -14.5, -23., -14.5, -14.5, -12.],
                                   [133., 2750., 159.5, 159.5, 133., 159.5, 159.5, 132.]])
                                   #发电机变量上下限
    YawSystemLimits = np.array([[-25., 0., -900., -216.],
                                   [275, 120., 900., 216.]]) #偏航系统变量上下限
    PowerLimits = np.array([[-250., -250., -250., 225., 225., 225., -1., -1.2, 0., -1250., -305.],
                               [2750., 2750., 2750., 525., 525., 525., 1., 1.2, 1., 1750., 2755.]])
                                #电网变量上下限
    ConverterLimits = np.array([[-12.5, 0.],
                                   [77.5, 100.]]) #变流器变量上下限
    FanBrakeLimits = np.array([[0., 0.],
                                  [70., 9.]]) #风机制动变量上下限
    limits = [WeatherLimits, PitchSystemLimits, EgineRoomLimits, ControlVariableLimits,
                      GearBoxLimits, GeneratorLimits, YawSystemLimits, PowerLimits,
                      ConverterLimits, FanBrakeLimits]
    limits2 = SetData(limits)
    return limits2

def SetData(insset):
    Weather = insset[0]
    PitchSystem = insset[1]
    EgineRoom = insset[2]
    ControlVariable = insset[3]
    GearBox = insset[4]
    Generator = insset[5]
    YawSystem = insset[6]
    Power = insset[7]
    Converter = insset[8]
    FanBrake = insset[9]
    ins = np.hstack((Weather, PitchSystem, EgineRoom))
    ins = np.hstack((ins, ControlVariable, GearBox))
    ins = np.hstack((ins, Generator, YawSystem, Power))
    ins = np.hstack((ins, Converter, FanBrake))
    return ins

def Load_data(path, fault_code, epsilon):
    Weather_index = {}.fromkeys(['gWindSpeed10s', 'gWindSpeed1', 'gWindSpeed2',
                           'gWindDirection1Avg25s', 'gWindDirection2Avg25s'])
    Weather_append = np.zeros((1, 5))
    '''
    天气0：10秒钟平均风速；风速计1风速测量值；风速2风速测量值;
          风向标1风向测量值25秒平均值; 风向标2风向测量值25秒平均值
    '''
    PitchSystem_index = {}.fromkeys(['gRotSpeed1', 'gRotSpeed2', 'gRotSpeedSSI',
                               'gPitSysTagPosSet', 'gPitchPosition',
                               'gPitPosEnc1[0]', 'gPitPosEnc1[1]', 'gPitPosEnc1[2]',
                               'gPitPosEnc2[0]', 'gPitPosEnc2[1]', 'gPitPosEnc2[2]',
                   'gPitAxisBattVolt[0]', 'gPitAxisBattVolt[1]', 'gPitAxisBattVolt[2]',
                      'gbufBattBoxTemp[0]', 'gbufBattBoxTemp[1]', 'gbufBattBoxTemp[2]',
                       'gbufAxisBoxTemp[0]', 'gbufAxisBoxTemp[1]','gbufAxisBoxTemp[2]',
                            'gbufMotorTemp[0]', 'gbufMotorTemp[1]', 'gbufMotorTemp[2]',
                        'gbufHubTemp', 'gPitSpeed[0]', 'gPitSpeed[1]', 'gPitSpeed[2]'])
    PitchSystem_append = np.zeros((1, 27))
    '''
    变桨系统1：模拟量风轮转速1；模拟量风轮转速2；风轮转速；
             设定桨距角位置；桨距角实际位置；
             桨叶1电机编码器角度；桨叶2电机编码器角度；桨叶3电机编码器角度；
             桨叶1冗余编码器角度；桨叶2冗余编码器角度；桨叶3冗余编码器角度；
             桨叶1电池电压；桨叶2电池电压；桨叶3电池电压；
             变桨电池柜1温度；变桨电池柜2温度；变桨电池柜3温度；
             变桨轴柜1温度；变桨轴柜2温度；变桨轴柜3温度；
             变桨电机1温度；变桨电机2温度；变桨电机3温度；
             轮毂温度；变桨1速度；变桨2速度；变桨3速度
    '''
    EgineRoom_index = {}.fromkeys(['gNacVibrationX', 'gNacVibrationY', 'gVibEffectiveValue'])
    EgineRoom_append = np.zeros((1, 3))
    '''
    机舱2：机舱X周方向振动；机舱Y轴方向振动；机舱振动有效值
    '''
    ControlVariable_index = {}.fromkeys(['gTmpCabTb', 'gTmpTb', 'gTmpCabNac',
                                   'gTmpNac', 'gTmpNacOutdoor'])
    ControlVariable_append = np.zeros((1, 5))
    '''
    控制因素3：主控塔底柜柜内温度；风机塔底温度；风机机舱主控柜温度；
             风机机舱温度；风机舱位温度
    '''
    GearBox_index = {}.fromkeys(['gTmpGbxHigh', 'gTmpGbxLow', 'gTmpGbxInletOil',
                           'gTmpGbxOil', 'gTmpGbxWater'])
    GearBox_append = np.zeros((1, 5))
    '''
    齿轮箱4：齿轮箱高速轴温度；齿轮箱低速轴温度；齿轮箱进油口油温；
           齿轮箱油温；齿轮箱冷却水温度
    '''
    Generator_index = {}.fromkeys(['gActualGenSpd', 'gTmpGenWinding',
                             'gTmpGenWindingU1', 'gTmpGenWindingV1', 'gTmpGenWindingW1',
                             'gTmpGenBearingF','gTmpGenBearingR', 'gTmpGenRing'])
    Generator_append = np.zeros((1, 8))
    '''
    发电机5：发电机转速；发电机绕组温度；
           发电机转子U1温度；发电机转子V1温度；发电机转子W1温度；
           发电机前轴承温度；发电机后轴承温度；发电机滑环温度；
    '''
    YawSystem_index = {}.fromkeys(['gYawDev25s', 'gYawTwistPosition', 'gHydSysPressure',
                             'gYawCode'])
    YawSystem_append = np.zeros((1, 4))
    '''
    偏航系统6：偏航角度；扭缆位置；液压系统压力；风机偏航程序等级
    '''
    Power_index = {}.fromkeys(['gSCACosPhiSet', 'gScadaPowReduct',
                         'gGridU1', 'gGridU2', 'gGridU3',
                         'gGridIL1', 'gGridIL2', 'gGridIL3',
                         'gGridP', 'gGridQ', 'gGridCosPhi'])
    Power_append = np.zeros((1, 11))
    '''
    电网7：系统设定功率因数；系统设定功率工作点；
         电网A相电压；电网B相电压；电网C相电压；
         电网A相电流；电网B相电流；电网C相电流；
         有功功率；无功功率；功率因数
    '''
    Converter_index = {}.fromkeys(['gConvTorqueActualVal', 'gTmpConverterCoolWtr'])
    Converter_append = np.zeros((1, 2))
    '''
    变流器8：变流器实际输出转矩百分比；变流器冷却水温度
    '''
    FanBrake_index = {}.fromkeys(['gBrakeCode', 'gMainLoopNumber'])
    FanBrake_append = np.zeros((1, 2))
    '''
    风机制动9：风机制动等级；风机运行状态
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
            PitchSystem = np.zeros((1, 27))
            EgineRoom = np.zeros((1, 3))
            ControlVariable = np.zeros((1, 5))
            GearBox = np.zeros((1, 5))
            Generator = np.zeros((1, 8))
            YawSystem = np.zeros((1, 4))
            Power = np.zeros((1, 11))
            Converter = np.zeros((1, 2))
            FanBrake = np.zeros((1, 2))
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
                    if PitchSystem_index.has_key(k) == True:
                        PitchSystem_index[k] = v
                    if EgineRoom_index.has_key(k) == True:
                        EgineRoom_index[k] = v
                    if ControlVariable_index.has_key(k) == True:
                        ControlVariable_index[k] = v
                    if GearBox_index.has_key(k) == True:
                        GearBox_index[k] = v
                    if Generator_index.has_key(k) == True:
                        Generator_index[k] = v
                    if YawSystem_index.has_key(k) == True:
                        YawSystem_index[k] = v
                    if Power_index.has_key(k) == True:
                        Power_index[k] = v
                    if Converter_index.has_key(k) == True:
                        Converter_index[k] = v
                    if FanBrake_index.has_key(k) == True:
                        FanBrake_index[k] = v
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
                        for (k, v) in PitchSystem_index.items():
                            if v != None:
                                PitchSystem[line_index, j_1] = line[v]
                            j_1 += 1
                        j_2 = 0
                        for (k, v) in EgineRoom_index.items():
                            if v != None:
                                EgineRoom[line_index, j_2] = line[v]
                            j_2 += 1
                        j_3 = 0
                        for (k, v) in ControlVariable_index.items():
                            if v != None:
                                ControlVariable[line_index, j_3] = line[v]
                            j_3 += 1
                        j_4 = 0
                        for (k, v) in GearBox_index.items():
                            if v != None:
                                GearBox[line_index, j_4] = line[v]
                            j_4 += 1
                        j_5 = 0
                        for (k, v) in Generator_index.items():
                            if v != None:
                                Generator[line_index, j_5] = line[v]
                            j_5 += 1
                        j_6 = 0
                        for (k, v) in YawSystem_index.items():
                            if v != None:
                                YawSystem[line_index, j_6] = line[v]
                            j_6 += 1
                        j_7 = 0
                        for (k, v) in Power_index.items():
                            if v != None:
                                Power[line_index, j_7] = line[v]
                            j_7 += 1
                        j_8 = 0
                        for (k, v) in Converter_index.items():
                            if v != None:
                                Converter[line_index, j_8] = line[v]
                            j_8 += 1
                        j_9 = 0
                        for (k, v) in FanBrake_index.items():
                            if v != None:
                                FanBrake[line_index, j_9] = line[v]
                            j_9 += 1
                        line_index += 1
                        TimeStamp_0 = TimeStamp_1
                        Weather = np.vstack((Weather, Weather_append))
                        PitchSystem = np.vstack((PitchSystem, PitchSystem_append))
                        EgineRoom = np.vstack((EgineRoom, EgineRoom_append))
                        ControlVariable = np.vstack((ControlVariable, ControlVariable_append))
                        GearBox = np.vstack((GearBox, GearBox_append))
                        Generator = np.vstack((Generator, Generator_append))
                        YawSystem = np.vstack((YawSystem, YawSystem_append))
                        Power = np.vstack((Power, Power_append))
                        Converter = np.vstack((Converter, Converter_append))
                        FanBrake = np.vstack((FanBrake, FanBrake_append))
                        ms_list.append(ms)
            Weather = np.delete(Weather, -1, axis = 0)
            PitchSystem = np.delete(PitchSystem, -1, axis = 0)
            EgineRoom = np.delete(EgineRoom, -1, axis = 0)
            ControlVariable = np.delete(ControlVariable, -1, axis = 0)
            GearBox = np.delete(GearBox, -1, axis = 0)
            Generator = np.delete(Generator, -1, axis = 0)
            YawSystem = np.delete(YawSystem, -1, axis = 0)
            Power = np.delete(Power, -1, axis = 0)
            Converter = np.delete(Converter, -1, axis = 0)
            FanBrake = np.delete(FanBrake, -1, axis = 0)

            fault_level2 = np.array(fault_code)
            insset = [Weather, PitchSystem, EgineRoom, ControlVariable,GearBox,
                      Generator, YawSystem, Power, Converter, FanBrake]
            #insset_new = data_preprocessing(insset, epsilon)
            #insset_new = data_preprocessing3(insset)
            insset_new = SetData(insset)
            outsset = np.hstack((outsset,fault_level2)).astype('int32')
            insset_list.append(insset_new)
            fault_time.append(ft)
            msset.append(ms_list)
    return insset_list, outsset, fault_time, msset

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

def trainset_trans(dataset):
    x = dataset[0]
    y = dataset[1]
    x0 = x[0]
    y0 = y[0] * np.ones(x0.shape[0], dtype=int32)
    for i in range(len(x)):
        if  (i != 0):
            x1 = x[i]
            y1 = y[i] * np.ones(x1.shape[0], dtype=int32)
            x0 = np.vstack((x0, x1))
            y0 = np.hstack((y0, y1))
    return x0, y0

def load_mulitdata(path, epsilon):
    start_time = timeit.default_timer()
    path = os.path.expanduser(path)
    fault_code = []
    for (dirname, subdir, subfile) in os.walk(path):
        fault_num = 0
        for f1 in subdir:
            fault_code.append(f1)
            filename = os.path.join(dirname, f1)
            print('\n(' + filename + ')')
            if fault_num == 0:
                dataset = Load_data(filename, fault_num, epsilon)
            else:
                dataset = Add_data(dataset, filename, fault_num, epsilon)
            fault_num += 1
    ins, labels, fault_time, millisecond = dataset
    end_time = timeit.default_timer()
    print ('load dataset ran for %.2fmin' %((end_time - start_time)/60.))
    return ins, labels, fault_code, fault_time, millisecond

def load_binarydata(path, epsilon):
    start_time = timeit.default_timer()
    path = os.path.expanduser(path)
    fault_code = []
    for (dirname, subdir, subfile) in os.walk(path):
        fault_num = 0
        for f1 in subdir:
            fault_code.append(f1)
            filename = os.path.join(dirname, f1)
            print('\n(' + filename + ')')
            if fault_num == 0:
                dataset = Load_data(filename, f1, epsilon)
            else:
                dataset = Add_data(dataset, filename, f1, epsilon)
            fault_num += 1
    ins, labels, fault_time, millisecond = dataset
    end_time = timeit.default_timer()
    print ('load dataset ran for %.2fmin' %((end_time - start_time)/60.))
    return ins, labels, fault_code, fault_time, millisecond

def feature_segment(dataset):
    weatherset = []
    setpointset = []
    variablesset = []
    for i in range(len(dataset)):
        weather = dataset[i][:, : 5]
        weatherext = np.vstack((0.4*weather[:,0]+0.4*weather[:,2]+0.2*weather[:,3],
                                0.5*weather[:,1]+0.5*weather[:,4]))
        weatherext = transpose(weatherext)
        setpoint = np.vstack((dataset[i][:, 16], dataset[i][:, 54], dataset[i][:, 63],
                              dataset[i][:, 65], dataset[i][:, 70], dataset[i][:, 71]))
        setpoint = transpose(setpoint)
        variables = np.hstack((dataset[i][:, 5:16], dataset[i][:, 17:54], dataset[i][:, 55:63],
                                dataset[i][:, 64:65], dataset[i][:, 66:70]))
        weatherset.append(weatherext)
        setpointset.append(setpoint)
        variablesset.append(variables)
    return weatherset, setpointset, variablesset

def Data_segment(path, path_w, path_sp, path_v):
    dataset = np.load(path)
    wraw, spraw, vraw = feature_segment(dataset)
    np.save(path_w, wraw)
    np.save(path_sp, spraw)
    np.save(path_v, vraw)
    return wraw, spraw, vraw

def variables_segment(path):
    dataset = np.load(path)
    v1set = []
    v2set = []
    for i in range(len(dataset)):
        v1 = np.hstack((dataset[i][:, 0:1], dataset[i][:, 2:5], dataset[i][:, 7:8],
                        dataset[i][:, 9:13], dataset[i][:, 14:15], dataset[i][:, 17:18],
                        dataset[i][:, 21:22], dataset[i][:, 25:29], dataset[i][:, 40:41],
                        dataset[i][:, 48:53], dataset[i][:, 56:59]))
        v2 = np.hstack((dataset[i][:, 1:2], dataset[i][:, 5:7], dataset[i][:, 8:9],
                        dataset[i][:, 13:14], dataset[i][:, 15:17], dataset[i][:, 18:21],
                        dataset[i][:, 22:25], dataset[i][:, 29:40], dataset[i][:, 41:48],
                        dataset[i][:, 53:56], dataset[i][:, 59:61]))
        v1set.append(v1)
        v2set.append(v2)
    np.save('data/codedata/80001_v1_train', v1set)
    np.save('data/codedata/80001_v2_train', v2set)
    return v1set, v2set

def variables_conb(v1, v2):
    v = []
    for i in range(len(v1)):
        v.append(np.hstack((v2[i], v1[i])))
    return v

def x_conb(w, sp, v):
    x = []
    for i in range(len(w)):
        x.append(np.hstack((w[i], sp[i], v[i])))
    return x

def samples_nor(x, limits):
    x_nor = minmax_standardization5(x, limits[0], limits[1])
    return x_nor

def samples_tnor(x, limits):
    xlist_nor = []
    for i in range(len(x)):
        x_nor = samples_nor(x[i], limits)
        xlist_nor.append(x_nor)
    return xlist_nor

def dataset_trans(dataset):
    x = dataset
    x0 = x[0]
    for i in range(len(x)):
        if  (i != 0):
            x1 = x[i]
            x0 = np.vstack((x0, x1))
    return x0

def samples_zca(x, x_mean, x_std, u, s, epsilon):
    x_fea = fea_standardization(x, x_mean, x_std)
    x_zca = zca_whitening(x_fea, u, s, epsilon)
    x_zca = fea_stand_inverse(x, x_mean, x_std)
    return x_zca

def samples_tzca(x, x_mean, x_std, u, s, epsilon):
    xlist_zca = []
    for i in range(len(x)):
        x_zca = samples_zca(x[i], x_mean, x_std, u, s, epsilon)
        xlist_zca.append(x_zca)
    return xlist_zca

def main():
    #path = 'data/codedata/x_train.npy'
    #path = '../../20180502/data/10+/80001_49/train'
    epsilon = 0.01
    #x_test, y_test, fault_code, ft_test, ms_test = load_binarydata(path, epsilon)
    #np.save('data/codedata/80001_x_train', np.delete(x_test, [9, 37], axis=0))
    #np.save('data/codedata/80001_y_train', np.delete(y_test, [9, 37], axis=0))
    #np.save('data/codedata/labeled_ft_train', ft_test)
    #np.save('data2/codedata/labeled_ms_train', ms_test)
    #print len(np.delete(x_test, [9, 37], axis=0)), len(np.delete(y_test, [9, 37], axis=0))
    #print np.delete(y_test, [9, 37], axis=0), np.delete(x_test, [9, 37], axis=0)[9], x_test[9], x_test[37]

    #path_raw = 'data/codedata/80001_x_train.npy'
    #path_w = 'data/codedata/80001_w_train'
    #path_sp = 'data/codedata/80001_sp_train'
    #path_v = 'data/codedata/80001_v_train.npy'
    #wraw, spraw, vraw = Data_segment(path_raw, path_w, path_sp, path_v)
    #print len(wraw), len(spraw), len(vraw)
    #print wraw[0].shape, spraw[0].shape, vraw[0].shape
    #v1raw, v2raw = variables_segment(path_v)
    #print len(v1raw), len(v2raw), v1raw[0].shape, v2raw[0].shape

    #limits = np.loadtxt('data/v2_limits.csv', delimiter=',')
    #raw = np.load('data/codedata/80001_v2_train.npy')
    #nor = samples_tnor(raw, limits)
    #np.save('data/codedata/80001_v2_nor_train', nor)
    #print len(nor), nor[0].shape

    #v1 = np.load('data/codedata/80001_v1_nor_train.npy')
    #v2 = np.load('data/codedata/80001_v2_nor_train.npy')
    #v = variables_conb(v1, v2)
    #print v1[0].shape, v2[0].shape
    #print len(v), v[0].shape
    #np.save('data/codedata/80001_v_nor_train.npy', v)

    #vnor = list(np.load('data2/codedata/labeled_vnor_train.npy')) + list(np.load('data2/codedata/unlabeled_vnor_train.npy'))
    #vnor = np.load('data/codedata/80001_v_nor_train.npy')
    #print len(vnor)
    #vtrans = dataset_trans(vnor)
    #print vtrans.shape
    #v_mean = np.mean(vtrans, axis=0)
    #v_std = np.std(vtrans, axis=0)
    #v_u, v_s = get_usv(vtrans, v_mean, v_std)
    #np.savetxt('data/v_mean.csv', v_mean, delimiter=',')
    #np.savetxt('data/v_std.csv', v_std, delimiter=',')
    #np.savetxt('data/v_u.csv', v_u, delimiter=',')
    #np.savetxt('data/v_s.csv', v_s, delimiter=',')

    #v_mean = np.loadtxt('data/v_mean.csv', delimiter=',')
    #v_std = np.loadtxt('data/v_std.csv', delimiter=',')
    #v_u = np.loadtxt('data/v_u.csv', delimiter=',')
    #v_s = np.loadtxt('data/v_s.csv', delimiter=',')
    #vnor = np.load('data/codedata/80001_v_nor_train.npy')
    #print v_mean.shape, v_std.shape, v_u.shape, v_s.shape, vnor.shape
    #vnortrans = dataset_trans(vnor)
    #vzca = samples_tzca(vnor, v_mean, v_std, v_u, v_s, epsilon)
    #np.save('data/codedata/80001_v_zca_train.npy', vzca)
    #print len(vzca), vzca[0].shape, vzca[0]

    #wnor = np.load('../data/expdata/w_nor_train.npy')
    #spnor = np.load('../data/expdata/sp_nor_train.npy')
    #vzca = np.load('../data/expdata/v_zca_train.npy')
    #print len(wnor), len(spnor), len(vzca)
    #xzca = x_conb(wnor, spnor, vzca)
    #print len(xzca), xzca[0].shape
    #np.save('../data/expdata/x_zca_train.npy', xzca)

    #xzca = list(np.load('data2/codedata/labeled_xzca_train.npy')) + list(np.load('data2/codedata/unlabeled_xzca_train.npy'))
    #print len(xzca)
    #xtrans = dataset_trans(xzca)


    '''
    y_label = np.load('../paper_test/paper_test/process4/labeled_y_train.npy')
    y_unlabel = np.load('../paper_test/paper_test/process4/unlabeled_y_train.npy')
    y = np.hstack((y_label, y_unlabel))
    y2 = np.load('../paper_test/paper_test/process4/Y_new_svt.npy')
    for i in range(len(y)):
        print int(y[i]), y2[i]
    '''

if __name__ == '__main__':
    main()
