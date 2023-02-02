# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 17:13:24 2023

@author: siliconsynapse
"""

import autograd.numpy as np
import math 



def rot_x(theta):

    Rx = [[1, 0, 0],
          [0, math.cos(theta), -math.sin(theta)],
         [ 0, math.sin(theta), math.cos(theta)]]
    Rx=np.array(Rx)
    return Rx


def rot_y(theta):
    Ry = [ [math.cos(theta), 0, math.sin(theta)], 
          [0, 1, 0],
          [-math.sin(theta), 0, math.cos(theta)]]
    Ry=np.array(Ry)
    return Ry


def rot_z(theta):
    Rz = [[math.cos(theta), -math.sin(theta), 0],
          [math.sin(theta), math.cos(theta), 0],
          [0, 0, 1]]
    Rz=np.array(Rz)
    return Rz

def skew(v):
    S = [[0, -v[2], v[1]] ,[v[2], 0, -v[0]],[-v[1], v[0], 0]]
    S=np.array(S)
    return S


def  func_feet_end_Jacobians(in1,in2):
    
    #FUNC_FEET_END_JACOBIANS
    #    [J_FR,J_FL,J_HR,J_HL] = FUNC_FEET_END_JACOBIANS(IN1,IN2)
    
    #    This function was generated by the Symbolic Math Toolbox version 8.4.
    #    24-Jul-2020 14:37:26
    
    Rb_1_1 = in1[15,:]
    Rb_1_2 = in1[18,:]
    Rb_1_3 = in1[21,:]
    Rb_2_1 = in1[16,:]
    Rb_2_2 = in1[19,:]
    Rb_2_3 = in1[22,:]
    Rb_3_1 = in1[17,:]
    Rb_3_2 = in1[20,:]
    Rb_3_3 = in1[23,:]
    qf_1 = in1[0,:]
    qf_2 = in1[1,:]
    qf_3 = in1[2,:]
    qf_4 = in1[3,:]
    qs_1 = in1[4,:]
    qs_2 = in1[5,:]
    qs_3 = in1[6,:]
    qs_4 = in1[7,:]
    t2 = math.cos(qf_1)
    t3 = math.cos(qf_2)
    t4 = math.cos(qf_3)
    t5 = math.cos(qf_4)
    t6 = math.cos(qs_1)
    t7 = math.cos(qs_2)
    t8 = math.cos(qs_3)
    t9 = math.cos(qs_4)
    t10 = math.sin(qf_1)
    t11 = math.sin(qf_2)
    t12 = math.sin(qf_3)
    t13 = math.sin(qf_4)
    t14 = math.sin(qs_1)
    
   
    
    J_FR = np.reshape([np.dot(Rb_1_1,t6)- np.dot(Rb_1_3,np.dot(t2,t14)) + np.dot(Rb_1_2,np.dot(t10,t14)), np.dot(Rb_2_1,t6)-np.dot(Rb_2_3,np.dot(t2,t14)) + np.dot(Rb_2_2,np.dot(t10,t14)), np.dot(Rb_3_1,t6)-np.dot(Rb_3_3,np.dot(t2,t14))+ np.dot(Rb_3_2,np.dot(t10,t14)), np.dot(Rb_1_2,t2)+ np.dot(Rb_1_3,t10), np.dot(Rb_2_2,t2)+ np.dot(Rb_2_3,t10), np.dot(Rb_3_2,t2) + np.dot(Rb_3_3,t10), np.dot(-Rb_1_1,t14)-np.dot(Rb_1_3,np.dot(t2,t6))+ np.dot(Rb_1_2,np.dot(t6,t10)),np.dot(-Rb_2_1,t14)-np.dot(Rb_2_3,np.dot(t2,t6))+ np.dot(Rb_2_2,np.dot(t6,t10)), np.dot(-Rb_3_1,t14)-np.dot(Rb_3_3,np.dot(t2,t6))+ np.dot(Rb_3_2,np.dot(t6,t10))],(3,3),order='F')
    
    t15 = math.sin(qs_2)
    J_FL = np.reshape([np.dot(Rb_1_1,t7)-np.dot(Rb_1_3,np.dot(t3,t15))+np.dot(Rb_1_2,np.dot(t11,t15)), np.dot(Rb_2_1,t7)- np.dot(Rb_2_3,np.dot(t3,t15))+np.dot(Rb_2_2,np.dot(t11,t15)),np.dot(Rb_3_1,t7)-np.dot(Rb_3_3,np.dot(t3,t15))+ np.dot(Rb_3_2, np.dot(t11,t15)),np.dot(Rb_1_2,t3)+ np.dot(Rb_1_3,t11), np.dot(Rb_2_2,t3)+ np.dot(Rb_2_3,t11),np.dot(Rb_3_2,t3)+np.dot(Rb_3_3,t11),-np.dot(Rb_1_1,t15)- np.dot(Rb_1_3,np.dot(t3,t7))+ np.dot(Rb_1_2,np.dot(t7,t11)),-np.dot(Rb_2_1,t15)-np.dot(Rb_2_3,np.dot(t3,t7))+ np.dot(Rb_2_2,np.dot(t7,t11)),-np.dot(Rb_3_1,t15)-np.dot(Rb_3_3,np.dot(t3,t7))+np.dot(Rb_3_2,np.dot(t7,t11))],(3,3),order='F')
    
    
    t16 = math.sin(qs_3)
    J_HR = np.reshape([np.dot(Rb_1_1,t8)-np.dot(Rb_1_3,np.dot(t4,t16))+ np.dot(Rb_1_2,np.dot(t12,t16)), np.dot(Rb_2_1,t8)- np.dot(Rb_2_3,np.dot(t4,t16))+ np.dot(Rb_2_2,np.dot(t12,t16)), np.dot(Rb_3_1,t8)- np.dot(Rb_3_3,np.dot(t4,t16))+ np.dot(Rb_3_2,np.dot(t12,t16)), np.dot(Rb_1_2,t4)+ np.dot(Rb_1_3,t12), np.dot(Rb_2_2,t4)+ np.dot(Rb_2_3,t12), np.dot(Rb_3_2,t4)+ np.dot(Rb_3_3,t12),-np.dot(Rb_1_1,t16)-np.dot(Rb_1_3,np.dot(t4,t8))+ np.dot(Rb_1_2,np.dot(t8,t12)),-np.dot(Rb_2_1,t16)-np.dot(Rb_2_3,np.dot(t4,t8))+ np.dot(Rb_2_2,np.dot(t8,t12)),-np.dot(Rb_3_1,t16)-np.dot(Rb_3_3,np.dot(t4,t8))+np.dot(Rb_3_2,np.dot(t8,t12))],(3,3),order='F')
    
    
    t17 = math.sin(qs_4)
    J_HL = np.reshape([np.dot(Rb_1_1,t9)-np.dot(Rb_1_3,np.dot(t5,t17))+ np.dot(Rb_1_2,np.dot(t13,t17)), np.dot(Rb_2_1,t9)-np.dot(Rb_2_3,np.dot(t5,t17))+ np.dot(Rb_2_2,np.dot(t13,t17)), np.dot(Rb_3_1,t9)- np.dot(Rb_3_3,np.dot(t5,t17))+ np.dot(Rb_3_2,np.dot(t13,t17)),np.dot(Rb_1_2,t5)+ np.dot(Rb_1_3,t13), np.dot(Rb_2_2,t5)+ np.dot(Rb_2_3,t13), np.dot(Rb_3_2,t5)+ np.dot(Rb_3_3,t13),-np.dot(Rb_1_1,t17)-np.dot(Rb_1_3,np.dot(t5,t9))+ np.dot(Rb_1_2,np.dot(t9,t13)),-np.dot(Rb_2_1,t17)-np.dot(Rb_2_3,np.dot(t5,t9))+ np.dot(Rb_2_2,np.dot(t9,t13)),-np.dot(Rb_3_1,t17)-np.dot(Rb_3_3,np.dot(t5,t9))+ np.dot(Rb_3_2,np.dot(t9,t13))],(3,3),order='F')
    
    return J_FR , J_FL , J_HR, J_HL



def func_feet(in1,in2):
    #FUNC_FEET
    #   [POS_FEET,VEL_FEET] = FUNC_FEET(IN1,IN2)
    
    #    This function was generated by the Symbolic Math Toolbox version 8.4.
    #    24-Jul-2020 14:37:36
    
    Rb_1_1 = in1[15,:]
    Rb_1_2 = in1[18,:]
    Rb_1_3 = in1[21,:]
    Rb_2_1 = in1[16,:]
    Rb_2_2 = in1[19,:]
    Rb_2_3 = in1[22,:]
    Rb_3_1 = in1[17,:]
    Rb_3_2 = in1[20,:]
    Rb_3_3 = in1[23,:]
    lh_FL_1 = in2[8,:]
    lh_FL_2 = in2[9,:]
    lh_FL_3 = in2[10,:]
    lh_HL_1 = in2[14,:]
    lh_HL_2 = in2[15,:]
    lh_HL_3 = in2[16,:]
    lh_FR_1 = in2[5,:]
    lh_FR_2 = in2[6,:]
    lh_FR_3 = in2[7,:]
    lh_HR_1 = in2[11,:]
    lh_HR_2 = in2[12,:]
    lh_HR_3 = in2[13,:]
    qcx_1 = in1[42,:]
    qcx_2 = in1[43,:]
    qcx_3 = in1[44,:]
    qcx_4 = in1[45,:]
    qcxD_1 = in1[50,:]
    qcxD_2 = in1[51,:]
    qcxD_3 = in1[52,:]
    qcxD_4 = in1[53,:]
    qcy_1 = in1[46,:]
    qcy_2 = in1[47,:]
    qcy_3 = in1[48,:]
    qcy_4 = in1[49,:]
    qcyD_1 = in1[54,:]
    qcyD_2 = in1[55,:]
    qcyD_3 = in1[56,:]
    qcyD_4 = in1[57,:]
    qfD_1 = in1[24,:]
    qfD_2 = in1[25,:]
    qfD_3 = in1[26,:]
    qfD_4 = in1[27,:]
    qf_1 = in1[0,:]
    qf_2 = in1[1,:]
    qf_3 = in1[2,:]
    qf_4 = in1[3,:]
    qlD_1 = in1[32,:]
    qlD_2 = in1[33,:]
    qlD_3 = in1[34,:]
    qlD_4 = in1[35,:]
    ql_1 = in1[8,:]
    ql_2 = in1[9,:]
    ql_3 = in1[10,:]
    ql_4 = in1[11,:]
    qsD_1 = in1[28,:]
    qsD_2 = in1[29,:]
    qsD_3 = in1[30,:]
    qsD_4 = in1[31,:]
    qs_1 = in1[4,:]
    qs_2 = in1[5,:]
    qs_3 = in1[6,:]
    qs_4 = in1[7,:]
    vb_1 = in1[36,:]
    vb_2 = in1[37,:]
    vb_3 = in1[38,:]
    wb_1 = in1[39,:]
    wb_2 = in1[40,:]
    wb_3 = in1[41,:]
    xb_1 = in1[12,:]
    xb_2 = in1[13,:]
    xb_3 = in1[14,:]
    t2 = math.cos(qf_1)
    t3 = math.cos(qf_2)
    t4 = math.cos(qf_3)
    t5 = math.cos(qf_4)
    t6 = math.cos(qs_1)
    t7 = math.cos(qs_2)
    t8 = math.cos(qs_3)
    t9 = math.cos(qs_4)
    t10 = math.sin(qf_1)
    t11 = math.sin(qf_2)
    t12 = math.sin(qf_3)
    t13 = math.sin(qf_4)
    t14 = math.sin(qs_1)
    t15 = math.sin(qs_2)
    t16 = math.sin(qs_3)
    t17 = math.sin(qs_4)
    t18 = np.dot(Rb_1_1,wb_2)
    t19 = np.dot(Rb_1_2,wb_1)
    t20 = np.dot(Rb_1_1,wb_3)
    t21 = np.dot(Rb_1_3,wb_1)
    t22 = np.dot(Rb_1_2,wb_3)
    t23 = np.dot(Rb_1_3,wb_2)
    t24 = np.dot(Rb_2_1,wb_2)
    t25 = np.dot(Rb_2_2,wb_1)
    t26 = np.dot(Rb_2_1,wb_3)
    t27 = np.dot(Rb_2_3,wb_1)
    t28 = np.dot(Rb_2_2,wb_3)
    t29 = np.dot(Rb_2_3,wb_2)
    t30 = np.dot(Rb_3_1,wb_2)
    t31 = np.dot(Rb_3_2,wb_1)
    t32 = np.dot(Rb_3_1,wb_3)
    t33 = np.dot(Rb_3_3,wb_1)
    t34 = np.dot(Rb_3_2,wb_3)
    t35 = np.dot(Rb_3_3,wb_2)
    t36 = np.dot(qcy_1,t2)
    t37 = np.dot(qcy_2,t3)
    t38 = np.dot(qcy_3,t4)
    t39 = np.dot(qcy_4,t5)
    t40 = np.dot(qcyD_1,t2)
    t41 = np.dot(qcyD_2,t3)
    t42 = np.dot(qcyD_3,t4)
    t43 = np.dot(qcyD_4,t5)
    t44 = np.dot(qcx_1,t6)
    t45 = np.dot(qcx_2,t7)
    t46 = np.dot(qcx_3,t8)
    t47 = np.dot(qcx_4,t9)
    t48 = np.dot(qcxD_1,t6)
    t49 = np.dot(qcxD_2,t7)
    t50 = np.dot(qcxD_3,t8)
    t51 = np.dot(qcxD_4,t9)
    t52 = np.dot(qcy_1,t10)
    t53 = np.dot(qcy_2,t11)
    t54 = np.dot(qcy_3,t12)
    t55 = np.dot(qcy_4,t13)
    t56 = np.dot(qcyD_1,t10)
    t57 = np.dot(qcyD_2,t11)
    t58 = np.dot(qcyD_3,t12)
    t59 = np.dot(qcyD_4,t13)
    t60 = np.dot(qlD_1,t14)
    t61 = np.dot(qlD_2,t15)
    t62 = np.dot(qlD_3,t16)
    t63 = np.dot(qlD_4,t17)
    t64 = np.dot(ql_1,t14)
    t65 = np.dot(ql_2,t15)
    t66 = np.dot(ql_3,t16)
    t67 = np.dot(ql_4,t17)
    t72 = np.dot(ql_1,np.dot(qsD_1,t6))
    t73 = np.dot(ql_2,np.dot(qsD_2,t7))
    t74 = np.dot(ql_3,np.dot(qsD_3,t8))
    t75 = np.dot(ql_4,np.dot(qsD_4,t9))
    t80 = np.dot(qcx_1,np.dot(qsD_1,t14))
    t81 = np.dot(qcx_2,np.dot(qsD_2,t15))
    t82 = np.dot(qcx_3,np.dot(qsD_3,t16))
    t83 = np.dot(qcx_4,np.dot(qsD_4,t17))
    t84 = -t19
    t85 = -t21
    t86 = -t23
    t87 = -t25
    t88 = -t27
    t89 = -t29
    t90 = -t31
    t91 = -t33
    t92 = -t35
    t93 = np.dot(qlD_1,np.dot(t2,t6))
    t94 = np.dot(qlD_2,np.dot(t3,t7))
    t95 = np.dot(qlD_3,np.dot(t4,t8))
    t96 = np.dot(qlD_4,np.dot(t5,t9))
    t97 = np.dot(ql_1,np.dot(t2,t6))
    t98 = np.dot(ql_2,np.dot(t3,t7))
    t99 = np.dot(ql_3,np.dot(t4,t8))
    t100 = np.dot(ql_4,np.dot(t5,t9))
    t101 = np.dot(qcx_1,np.dot(t2,t14))
    t102 = np.dot(qcx_2,np.dot(t3,t15))
    t103 = np.dot(qcx_3,np.dot(t4,t16))
    t104 = np.dot(qcx_4,np.dot(t5,t17))
    t105 = np.dot(qcxD_1,np.dot(t2,t14))
    t106 = np.dot(qcxD_2,np.dot(t3,t15))
    t107 = np.dot(qcxD_3,np.dot(t4,t16))
    t108 = np.dot(qcxD_4,np.dot(t5,t17))
    t109 = np.dot(qlD_1,np.dot(t6,t10))
    t110 = np.dot(qlD_2,np.dot(t7,t11))
    t111 = np.dot(qlD_3,np.dot(t8,t12))
    t112 = np.dot(qlD_4,np.dot(t9,t13))
    t113 = np.dot(ql_1,np.dot(t6,t10))
    t114 = np.dot(ql_2,np.dot(t7,t11))
    t115 = np.dot(ql_3,np.dot(t8,t12))
    t116 = np.dot(ql_4,np.dot(t9,t13))
    t117 = np.dot(qcx_1,np.dot(t10,t14))
    t118 = np.dot(qcx_2,np.dot(t11,t15))
    t119 = np.dot(qcx_3,np.dot(t12,t16))
    t120 = np.dot(qcx_4,np.dot(t13,t17))
    t121 = np.dot(qcxD_1,np.dot(t10,t14))
    t122 = np.dot(qcxD_2,np.dot(t11,t15))
    t123 = np.dot(qcxD_3,np.dot(t12,t16))
    t124 = np.dot(qcxD_4,np.dot(t13,t17))
    t68 = np.dot(qfD_1,t36)
    t69 = np.dot(qfD_2,t37)
    t70 = np.dot(qfD_3,t38)
    t71 = np.dot(qfD_4,t39)
    t76 = np.dot(qfD_1,t52)
    t77 = np.dot(qfD_2,t53)
    t78 = np.dot(qfD_3,t54)
    t79 = np.dot(qfD_4,t55)
    t125 = -t48
    t126 = -t49
    t127 = -t50
    t128 = -t51
    t129 = -t64
    t130 = -t65
    t131 = -t66
    t132 = -t67
    t133 = np.dot(qfD_1,t117)
    t134 = np.dot(qfD_2,t118)
    t135 = np.dot(qfD_3,t119)
    t136 = np.dot(qfD_4,t120)
    t137 = np.dot(qsD_1,np.dot(t10,t64))
    t138 = np.dot(qsD_2,np.dot(t11,t65))
    t139 = np.dot(qsD_3,np.dot(t12,t66))
    t140 = np.dot(qsD_4,np.dot(t13,t67))
    t145 = np.dot(qfD_1,t97)
    t146 = np.dot(qfD_2,t98)
    t147 = np.dot(qfD_3,t99)
    t148 = np.dot(qfD_4,t100)
    t149 = np.dot(qsD_1,np.dot(t2,t44))
    t150 = np.dot(qsD_2,np.dot(t3,t45))
    t151 = np.dot(qsD_3,np.dot(t4,t46))
    t152 = np.dot(qsD_4,np.dot(t5,t47))
    t153 = np.dot(qfD_1,t101)
    t154 = np.dot(qfD_2,t102)
    t155 = np.dot(qfD_3,t103)
    t156 = np.dot(qfD_4,t104)
    t157 = np.dot(qfD_1,t113)
    t158 = np.dot(qfD_2,t114)
    t159 = np.dot(qfD_3,t115)
    t160 = np.dot(qfD_4,t116)
    t161 = np.dot(qsD_1,np.dot(t10,t44))
    t162 = np.dot(qsD_2,np.dot(t11,t45))
    t163 = np.dot(qsD_3,np.dot(t12,t46))
    t164 = np.dot(qsD_4,np.dot(t13,t47))
    t165 = np.dot(qsD_1,np.dot(t2,t64))
    t166 = np.dot(qsD_2,np.dot(t3,t65))
    t167 = np.dot(qsD_3,np.dot(t4,t66))
    t168 = np.dot(qsD_4,np.dot(t5,t67))
    t169 = -t93
    t170 = -t94
    t171 = -t95
    t172 = -t96
    t173 = -t97
    t174 = -t98
    t175 = -t99
    t176 = -t100
    t177 = -t101
    t178 = -t102
    t179 = -t103
    t180 = -t104
    t181 = -t105
    t182 = -t106
    t183 = -t107
    t184 = -t108
    t189 = t18+t84
    t190 = t20+t85
    t191 = t22+t86
    t192 = t24+t87
    t193 = t26+t88
    t194 = t28+t89
    t195 = t30+t90
    t196 = t32+t91
    t197 = t34+t92
    t206 = lh_FR_2+t36+t113+t117
    t207 = lh_FL_2+t37+t114+t118
    t208 = lh_HR_2+t38+t115+t119
    t209 = lh_HL_2+t39+t116+t120
    t141 = -t76
    t142 = -t77
    t143 = -t78
    t144 = -t79
    t185 = np.dot(qsD_1,np.dot(t10,t129))
    t186 = np.dot(qsD_2,np.dot(t11,t130))
    t187 = np.dot(qsD_3,np.dot(t12,t131))
    t188 = np.dot(qsD_4,np.dot(t13,t132))
    t198 = -t149
    t199 = -t150
    t200 = -t151
    t201 = -t152
    t202 = lh_FL_1+t45+t130
    t203 = lh_FR_1+t44+t129
    t204 = lh_HL_1+t47+t132
    t205 = lh_HR_1+t46+t131
    t210 = t60+t72+t80+t125
    t211 = t61+t73+t81+t126
    t212 = t62+t74+t82+t127
    t213 = t63+t75+t83+t128
    t214 = lh_FR_3+t52+t173+t177
    t215 = lh_FL_3+t53+t174+t178
    t216 = lh_HR_3+t54+t175+t179
    t217 = lh_HL_3+t55+t176+t180
    pos_feet = np.array([[xb_1 + np.dot(Rb_1_1,t203)+np.dot(Rb_1_2,t206)+ np.dot(Rb_1_3,t214)], [xb_2+np.dot(Rb_2_1,t203)+np.dot(Rb_2_2,t206)+np.dot(Rb_2_3,t214)],[xb_3+np.dot(Rb_3_1,t203)+np.dot(Rb_3_2,t206)+np.dot(Rb_3_3,t214)],[xb_1+np.dot(Rb_1_1,t202)+np.dot(Rb_1_2,t207)+np.dot(Rb_1_3,t215)],[xb_2+np.dot(Rb_2_1,t202)+np.dot(Rb_2_2,t207)+np.dot(Rb_2_3,t215)],[xb_3+np.dot(Rb_3_1,t202)+np.dot(Rb_3_2,t207)+np.dot(Rb_3_3,t215)],[xb_1+np.dot(Rb_1_1,t205)+np.dot(Rb_1_2,t208)+np.dot(Rb_1_3,t216)],[xb_2+np.dot(Rb_2_1,t205)+np.dot(Rb_2_2,t208)+np.dot(Rb_2_3,t216)],[xb_3+np.dot(Rb_3_1,t205)+np.dot(Rb_3_2,t208)+np.dot(Rb_3_3,t216)],[xb_1+np.dot(Rb_1_1,t204)+np.dot(Rb_1_2,t209)+np.dot(Rb_1_3,t217)],[xb_2+np.dot(Rb_2_1,t204)+np.dot(Rb_2_2,t209)+np.dot(Rb_2_3,t217)],[xb_3+np.dot(Rb_3_1,t204)+np.dot(Rb_3_2,t209)+np.dot(Rb_3_3,t217)]])
    pos_feet=pos_feet.reshape((12,1),order='F')
    t218 = t40+t109+t121+t141+t145+t153+t161+t185
    t219 = t41+t110+t122+t142+t146+t154+t162+t186
    t220 = t42+t111+t123+t143+t147+t155+t163+t187
    t221 = t43+t112+t124+t144+t148+t156+t164+t188
    t222 = t56+t68+t133+t157+t165+t169+t181+t198
    t223 = t57+t69+t134+t158+t166+t170+t182+t199
    t224 = t58+t70+t135+t159+t167+t171+t183+t200
    t225 = t59+t71+t136+t160+t168+t172+t184+t201
    vel_feet = np.array([[vb_1-np.dot(Rb_1_1,t210)+np.dot(Rb_1_2,t218)+np.dot(Rb_1_3,t222)+np.dot(t191,t203)-np.dot(t190,t206)+np.dot(t189,t214)],[vb_2-np.dot(Rb_2_1,t210)+np.dot(Rb_2_2,t218)+np.dot(Rb_2_3,t222)+np.dot(t194,t203)-np.dot(t193,t206)+np.dot(t192,t214)],[vb_3-np.dot(Rb_3_1,t210)+np.dot(Rb_3_2,t218)+np.dot(Rb_3_3,t222)+np.dot(t197,t203)-np.dot(t196,t206)+np.dot(t195,t214)],[vb_1-np.dot(Rb_1_1,t211)+np.dot(Rb_1_2,t219)+np.dot(Rb_1_3,t223)+np.dot(t191,t202)-np.dot(t190,t207)+np.dot(t189,t215)],[vb_2-np.dot(Rb_2_1,t211)+np.dot(Rb_2_2,t219)+np.dot(Rb_2_3,t223)+np.dot(t194,t202)-np.dot(t193,t207)+np.dot(t192,t215)],[vb_3-np.dot(Rb_3_1,t211)+np.dot(Rb_3_2,t219)+np.dot(Rb_3_3,t223)+np.dot(t197,t202)-np.dot(t196,t207)+np.dot(t195,t215)],[vb_1-np.dot(Rb_1_1,t212)+np.dot(Rb_1_2,t220)+np.dot(Rb_1_3,t224)+np.dot(t191,t205)-np.dot(t190,t208)+np.dot(t189,t216)],[vb_2-np.dot(Rb_2_1,t212)+np.dot(Rb_2_2,t220)+np.dot(Rb_2_3,t224)+np.dot(t194,t205)-np.dot(t193,t208)+np.dot(t192,t216)],[vb_3-np.dot(Rb_3_1,t212)+np.dot(Rb_3_2,t220)+np.dot(Rb_3_3,t224)+np.dot(t197,t205)-np.dot(t196,t208)+np.dot(t195,t216)],[vb_1-np.dot(Rb_1_1,t213)+np.dot(Rb_1_2,t221)+np.dot(Rb_1_3,t225)+np.dot(t191,t204)-np.dot(t190,t209)+np.dot(t189,t217)],[vb_2-np.dot(Rb_2_1,t213)+np.dot(Rb_2_2,t221)+np.dot(Rb_2_3,t225)+np.dot(t194,t204)-np.dot(t193,t209)+np.dot(t192,t217)],[vb_3-np.dot(Rb_3_1,t213)+np.dot(Rb_3_2,t221)+np.dot(Rb_3_3,t225)+np.dot(t197,t204)-np.dot(t196,t209)+np.dot(t195,t217)]])
    vel_feet=vel_feet.reshape((12,1),order='F')
    return pos_feet, vel_feet



def func_MhBe(in1,in2):
    #%FUNC_MHBE
    #%    [M,H,BG] = FUNC_MHBE(IN1,IN2)
    
    #%    This function was generated by the Symbolic Math Toolbox version 8.4.
    #%    24-Jul-2020 14:37:28
    
    Ib_1 = in2[2,:]
    Ib_2 = in2[3,:]
    Ib_3 = in2[4,:]
    Rb_1_1 = in1[15,:]
    Rb_1_2 = in1[18,:]
    Rb_1_3 = in1[21,:]
    Rb_2_1 = in1[16,:]
    Rb_2_2 = in1[19,:]
    Rb_2_3 = in1[22,:]
    Rb_3_1 = in1[17,:]
    Rb_3_2 = in1[20,:]
    Rb_3_3 = in1[23,:]
    g = in2[1,:]
    lh_FL_1 = in2[8,:]
    lh_FL_2 = in2[9,:]
    lh_FL_3 = in2[10,:]
    lh_HL_1 = in2[14,:]
    lh_HL_2 = in2[15,:]
    lh_HL_3 = in2[16,:]
    lh_FR_1 = in2[5,:]
    lh_FR_2 = in2[6,:]
    lh_FR_3 = in2[7,:]
    lh_HR_1 = in2[11,:]
    lh_HR_2 = in2[12,:]
    lh_HR_3 = in2[13,:]
    mb = in2[0,:]
    qcx_1 = in1[42,:]
    qcx_2 = in1[43,:]
    qcx_3 = in1[44,:]
    qcx_4 = in1[45,:]
    qcy_1 = in1[46,:]
    qcy_2 = in1[47,:]
    qcy_3 = in1[48,:]
    qcy_4 = in1[49,:]
    qf_1 = in1[0,:]
    qf_2 = in1[1,:]
    qf_3 = in1[2,:]
    qf_4 = in1[3,:]
    ql_1 = in1[8,:]
    ql_2 = in1[9,:]
    ql_3 = in1[10,:]
    ql_4 = in1[11,:]
    qs_1 = in1[4,:]
    qs_2 = in1[5,:]
    qs_3 = in1[6,:]
    qs_4 = in1[7,:]
    wb_1 = in1[39,:]
    wb_2 = in1[40,:]
    wb_3 = in1[41,:]
    M = np.reshape(np.block([mb,0.0,0.0,0.0,0.0,0.0,0.0,mb,0.0,0.0,0.0,0.0,0.0,0.0,mb,0.0,0.0,0.0,0.0,0.0,0.0,Ib_1,0.0,0.0,0.0,0.0,0.0,0.0,Ib_2,0.0,0.0,0.0,0.0,0.0,0.0,Ib_3]),[6,6],order='F')
   
    h = np.array([[0.0],[0.0],[np.dot(g,mb)],[-np.dot(Ib_2,np.dot(wb_2,wb_3))+np.dot(Ib_3,np.dot(wb_2,wb_3))],[np.dot(Ib_1,np.dot(wb_1,wb_3))-np.dot(Ib_3,np.dot(wb_1,wb_3))],-np.dot(Ib_1,np.dot(wb_1,wb_2))+np.dot(Ib_2,np.dot(wb_1,wb_2))],dtype=np.float32)
    
   
    t2 = math.cos(qf_1)
    t3 = math.cos(qf_2)
    t4 = math.cos(qf_3)
    t5 = math.cos(qf_4)
    t6 = math.cos(qs_1)
    t7 = math.cos(qs_2)
    t8 = math.cos(qs_3)
    t9 = math.cos(qs_4)
    t10 = math.sin(qf_1)
    t11 = math.sin(qf_2)
    t12 = math.sin(qf_3)
    t13 = math.sin(qf_4)
    t14 = math.sin(qs_1)
    t15 = math.sin(qs_2)
    t16 = math.sin(qs_3)
    t17 = math.sin(qs_4)
    t18 = np.dot(qcy_1,t2)
    t19 = np.dot(qcy_2,t3)
    t20 = np.dot(qcy_3,t4)
    t21 = np.dot(qcy_4,t5)
    t22 = np.dot(qcx_1,t6)
    t23 = np.dot(qcx_2,t7)
    t24 = np.dot(qcx_3,t8)
    t25 = np.dot(qcx_4,t9)
    t26 = np.dot(qcy_1,t10)
    t27 = np.dot(qcy_2,t11)
    t28 = np.dot(qcy_3,t12)
    t29 = np.dot(qcy_4,t13)
    t30 = np.dot(ql_1,t14);
    t31 = np.dot(ql_2,t15);
    t32 = np.dot(ql_3,t16);
    t33 = np.dot(ql_4,t17);
    t34 = np.dot(ql_1,np.dot(t2,t6))
    t35 = np.dot(ql_2,np.dot(t3,t7))
    t36 = np.dot(ql_3,np.dot(t4,t8))
    t37 = np.dot(ql_4,np.dot(t5,t9))
    t38 = np.dot(qcx_1,np.dot(t2,t14))
    t39 = np.dot(qcx_2,np.dot(t3,t15))
    t40 = np.dot(qcx_3,np.dot(t4,t16))
    t41 = np.dot(qcx_4,np.dot(t5,t17))
    t42 = np.dot(ql_1,np.dot(t6,t10))
    t43 = np.dot(ql_2,np.dot(t7,t11))
    t44 = np.dot(ql_3,np.dot(t8,t12))
    t45 = np.dot(ql_4,np.dot(t9,t13))
    t46 = np.dot(qcx_1,np.dot(t10,t14))
    t47 = np.dot(qcx_2,np.dot(t11,t15))
    t48 = np.dot(qcx_3,np.dot(t12,t16))
    t49 = np.dot(qcx_4,np.dot(t13,t17))
    t50 = -t30
    t51 = -t31
    t52 = -t32
    t53 = -t33
    t54 = -t34
    t55 = -t35
    t56 = -t36
    t57 = -t37
    t58 = -t38
    t59 = -t39
    t60 = -t40
    t61 = -t41
    t66 = lh_FR_2+t18+t42+t46
    t67 = lh_FL_2+t19+t43+t47
    t68 = lh_HR_2+t20+t44+t48
    t69 = lh_HL_2+t21+t45+t49
    t62 = lh_FL_1+t23+t51
    t63 = lh_FR_1+t22+t50
    t64 = lh_HL_1+t25+t53
    t65 = lh_HR_1+t24+t52
    t70 = lh_FR_3+t26+t54+t58
    t71 = lh_FL_3+t27+t55+t59
    t72 = lh_HR_3+t28+t56+t60
    t73 = lh_HL_3+t29+t57+t61
    Bg = np.reshape([1.0,0.0,0.0,np.dot(Rb_1_3,t66)-np.dot(Rb_1_2,t70),-np.dot(Rb_1_3,t63)+np.dot(Rb_1_1,t70),np.dot(Rb_1_2,t63)-np.dot(Rb_1_1,t66),0.0,1.0,0.0,np.dot(Rb_2_3,t66)-np.dot(Rb_2_2,t70),-np.dot(Rb_2_3,t63)+np.dot(Rb_2_1,t70),np.dot(Rb_2_2,t63)-np.dot(Rb_2_1,t66),0.0,0.0,1.0,np.dot(Rb_3_3,t66)-np.dot(Rb_3_2,t70),-np.dot(Rb_3_3,t63)+np.dot(Rb_3_1,t70),np.dot(Rb_3_2,t63)-np.dot(Rb_3_1,t66),1.0,0.0,0.0,np.dot(Rb_1_3,t67)-np.dot(Rb_1_2,t71),-np.dot(Rb_1_3,t62)+np.dot(Rb_1_1,t71),np.dot(Rb_1_2,t62)-np.dot(Rb_1_1,t67),0.0,1.0,0.0,np.dot(Rb_2_3,t67)-np.dot(Rb_2_2,t71),-np.dot(Rb_2_3,t62)+np.dot(Rb_2_1,t71),np.dot(Rb_2_2,t62)-np.dot(Rb_2_1,t67),0.0,0.0,1.0,np.dot(Rb_3_3,t67)-np.dot(Rb_3_2,t71),-np.dot(Rb_3_3,t62)+np.dot(Rb_3_1,t71),np.dot(Rb_3_2,t62)-np.dot(Rb_3_1,t67),1.0,0.0,0.0,np.dot(Rb_1_3,t68)-np.dot(Rb_1_2,t72),-np.dot(Rb_1_3,t65)+np.dot(Rb_1_1,t72),np.dot(Rb_1_2,t65)-np.dot(Rb_1_1,t68),0.0,1.0,0.0,np.dot(Rb_2_3,t68)-np.dot(Rb_2_2,t72),-np.dot(Rb_2_3,t65)+np.dot(Rb_2_1,t72),np.dot(Rb_2_2,t65)-np.dot(Rb_2_1,t68),0.0,0.0,1.0,np.dot(Rb_3_3,t68)-np.dot(Rb_3_2,t72),-np.dot(Rb_3_3,t65)+np.dot(Rb_3_1,t72),np.dot(Rb_3_2,t65)-np.dot(Rb_3_1,t68),1.0,0.0,0.0,np.dot(Rb_1_3,t69)-np.dot(Rb_1_2,t73),-np.dot(Rb_1_3,t64)+np.dot(Rb_1_1,t73),np.dot(Rb_1_2,t64)-np.dot(Rb_1_1,t69),0.0,1.0,0.0,np.dot(Rb_2_3,t69)-np.dot(Rb_2_2,t73),-np.dot(Rb_2_3,t64)+np.dot(Rb_2_1,t73),np.dot(Rb_2_2,t64)-np.dot(Rb_2_1,t69),0.0,0.0,1.0,np.dot(Rb_3_3,t69)-np.dot(Rb_3_2,t73),-np.dot(Rb_3_3,t64)+np.dot(Rb_3_1,t73),np.dot(Rb_3_2,t64)-np.dot(Rb_3_1,t69)],(12,6))
    Bg=Bg.T

    return M,h,Bg


