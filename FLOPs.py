

def attention_layer_FLOPs(B, N, D, D_in, D_h):
    output = 2 * D_in * N * N + 3 * D * D_in * N + N * D_h * D_in + N * D_h * D
    return B*output


def cross_attention_layer_FLOPs(Nq, Nk, Dq, Dk, D_in, D_h):
    output_CA = Nq * Dq * D_in + 2 * Nk * Dk * D_in + 2 * Nq * Nk * D_in + Nq * D_in *D_h + Nq * D_h * Dq
    output_SA = 3 * Nq * Dq * D_in + 2 * Nq * Nq * D_in
    return output_CA + output_SA

def cross_attention_layer_FLOPs_MR(Nq, Nk, Dq, Dk, D_in, D_h, DR):
    output_CA = Nq * Dq * D_in + 2 * Nk * Dk * D_in + 2 * Nq * Nk * D_in + Nq * D_in *D_h + Nq * D_h * Dq
    output_SA = 3 * Nq * Dq * D_in + 2 * Nq * Nq * D_in
    return output_CA*(1-DR) + output_SA



print("===========base===============")

stage0 = 2*attention_layer_FLOPs(1024, 64, 112, 336, 448)
stage1 = 3*attention_layer_FLOPs(1024, 16, 224, 672, 896)
stage2 = 13*attention_layer_FLOPs(25, 196, 448, 1344, 1792) + 3*attention_layer_FLOPs(1, 4096, 448, 1344, 1792)
total = (stage0+stage1+stage2)
stage2_WR = 0.25*(13*attention_layer_FLOPs(25, 196, 448, 1344, 1792) + 2*attention_layer_FLOPs(1, 4096, 448, 1344, 1792)) +attention_layer_FLOPs(1, 4096, 448, 1344, 1792)
total_WR = stage0+stage1+stage2_WR
print('stage0:', stage0/1e+9, stage0/total)
print('stage1:', stage1/1e+9, stage1/total)
print('stage2:', stage2/1e+9, stage2/total)
print('total:', total/1e+9)

print('stage2_WR:', stage2_WR/1e+9)
print('total_WR:', total_WR/1e+9)

print('FLOPs reduction:', (total)/total_WR)

print("===========large===============")

stage0 = 2*attention_layer_FLOPs(1024, 64, 144, 432, 576)
stage1 = 6*attention_layer_FLOPs(1024, 16, 288, 864, 1152)
stage2 = 33*attention_layer_FLOPs(16, 256, 576, 1728, 2304) + 3*attention_layer_FLOPs(1, 4096, 576, 1728, 2304)
total = (stage0+stage1+stage2)
stage2_WR = 0.25*(13*attention_layer_FLOPs(16, 256, 576, 1728, 2304) + 2*attention_layer_FLOPs(1, 4096, 576, 1728, 2304)) +attention_layer_FLOPs(1, 4096, 576, 1728, 2304)
total_WR = stage0+stage1+stage2_WR
print('stage0:', stage0/1e+9, stage0/total)
print('stage1:', stage1/1e+9, stage1/total)
print('stage2:', stage2/1e+9, stage2/total)
print('total:', total/1e+9)

print('stage2_WR:', stage2_WR/1e+9)
print('total_WR:', total_WR/1e+9)

print('FLOPs reduction:', (total)/total_WR)

print("===========MA===============")

MA_layers = 4*cross_attention_layer_FLOPs(4096, 4096*7, 256, 64, 256, 2048) 
MR_MA_layers = 4*cross_attention_layer_FLOPs_MR(4096, 4096*7, 256, 64, 256, 2048, 0.95*5/7) 

print('MA:', MA_layers/1e+9)
print('MR_MA:', MR_MA_layers/1e+9)
print('FLOPs reduction:', (MA_layers)/MR_MA_layers)

