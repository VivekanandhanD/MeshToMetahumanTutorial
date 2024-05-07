import face_alignment
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

#from skimage import io

def main() :
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device='cpu', face_detector='sfd')
    
    preds = fa.get_landmarks_from_directory('input')
    #print(preds)
    
    for i in preds:
        file_name = i.split("\\")[1].split(".")[0]
        print(file_name)
        out = preds[i][0]
    
        Lm3D = out
        lm_idx = np.array([31,37,40,43,46,49,55]) - 1
        Lm3D = np.stack([Lm3D[lm_idx[0],:],np.mean(Lm3D[lm_idx[[1,2]],:],0),np.mean(Lm3D[lm_idx[[3,4]],:],0),Lm3D[lm_idx[5],:],Lm3D[lm_idx[6],:]], axis = 0)
        Lm3D = Lm3D[[1,2,0,3,4],:]
        content = str(Lm3D).replace('[', '').replace(']', '')
        print(content)
        f = open('input\\detections\\' + file_name + '.txt', "w")
        f.write(content)
        f.close()
        
if __name__ == '__main__':
    main()
