import sys
import os
import scipy.io as sio
import cv2


path = os.path.dirname(__file__)
CENTERNET_PATH = 'D:/CenterMask/src/lib'#os.path.join(path,'../src/lib')
sys.path.insert(0, CENTERNET_PATH)

from detectors.detector_factory import detector_factory
from opts_pose import opts
from datasets.dataset_factory import get_dataset


def test_img(MODEL_PATH):
    debug = 1            # draw and show the result image
    TASK = 'ctdet'  
    input_h, intput_w = 800, 800
    opt = opts().init('--task {} --load_model {} --debug {} --input_h {} --input_w {}'.format(
        TASK, MODEL_PATH, debug, intput_w, input_h).split(' '))

    detector = detector_factory[opt.task](opt)

    img = '../readme/000388.jpg'
    ret = detector.run(img)['results']

def test_vedio(model_path, vedio_path=None):
    debug = -1            # return the result image with draw
    TASK = 'ctdet'  
    vis_thresh = 0.45
    input_h, intput_w = 800, 800
    opt = opts().init('--task {} --load_model {} --debug {} --input_h {} --input_w {} --vis_thresh {}'.format(
        TASK, MODEL_PATH, debug, intput_w, input_h, vis_thresh).split(' '))
    print(opt.dataset)
    detector = detector_factory[opt.task](opt)


    vedio = vedio_path if vedio_path else 0
    cap = cv2.VideoCapture(vedio, cv2.CAP_DSHOW)
    cap.set(3,1280)
    cap.set(4,760)
    while cap.isOpened():
        det = cap.grab()
        if det:
            flag, frame = cap.retrieve()
            res = detector.run(frame)
            cv2.imshow('face detect', res['plot_img'])

        if cv2.waitKey(1)&0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def test_wider_Face(model_path):
    Path = '../data/datasets/WIDERFACE/WIDER_val/images/'
    wider_face_mat = sio.loadmat('../data/datasets/WIDERFACE/wider_face_split/wider_face_val.mat')
    event_list = wider_face_mat['event_list']
    file_list = wider_face_mat['file_list']
    save_path = '../output/widerface/'
    save_path_img = '../output/widerface_img/'

    debug = 0          # return the detect result without show
    threshold = 0.05
    TASK = 'ctdet'#'ctdet'  
    input_h, intput_w = 512, 512
    opt = opts().init('--task {} --load_model {} --debug {} --vis_thresh {} --input_h {} --input_w {}'.format(
        TASK, MODEL_PATH, debug, threshold, input_h, intput_w).split(' '))
    detector = detector_factory[opt.task](opt)

    for index, event in enumerate(event_list):
        file_list_item = file_list[index][0]
        im_dir = event[0][0]
        if not os.path.exists(save_path + im_dir):
            #os.makedirs(save_path+'/mask/'+ im_dir)
            #os.makedirs(save_path+'/face/'+ im_dir)
            os.makedirs(save_path + im_dir)
            os.makedirs(save_path_img + im_dir)
        for num, file in enumerate(file_list_item):
            im_name = file[0][0]
            zip_name = '%s/%s.jpg' % (im_dir, im_name)
            print(os.path.join(Path, zip_name))
            img_path = os.path.join(Path, zip_name)
            orig_image = cv2.imread(img_path)
            dets= detector.run(img_path)['results']
            #print(dets)
            f = open(save_path + im_dir + '/' + im_name + '.txt', 'w')
            f.write('{:s}\n'.format('%s/%s.jpg' % (im_dir, im_name)))
            f.write('{:d}\n'.format(len(dets)))
            """
            ff = open(save_path+ '/mask/' + im_dir +'/' + im_name + '.txt', 'w')
            ff.write('{:s}\n'.format('%s/%s.jpg' % (im_dir, im_name)))
            ff.write('{:d}\n'.format(len(dets)))
            
            
            f = open(save_path  + '/face/'+ im_dir +'/'+ im_name + '.txt', 'w')
            f.write('{:s}\n'.format('%s/%s.jpg' % (im_dir, im_name)))
            f.write('{:d}\n'.format(len(dets)))
            """
            
            counts0 = 0
            counts1 = 0
            for b in dets[1]:
                x1, y1, x2, y2, s = b[0], b[1], b[2], b[3], b[4]
                f.write('{:.1f} {:.1f} {:.1f} {:.1f} {:.3f}\n'.format(x1, y1, (x2 - x1 + 1), (y2 - y1 + 1), s))
                if s>0.4:
                    #print(b)
                    counts0=counts0+1
                    cv2.rectangle(orig_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                    #cv2.circle(orig_image,(int(b[5]),int(b[6])),0,(0, 0, 255), 2)
                    #cv2.circle(orig_image,(int(b[7]),int(b[8])),0,(0, 0, 255), 2)
                    #cv2.putText(orig_image, str(0), (int(x1), int(y1 + 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
            """
            for b in dets[2]:
                x1, y1, x2, y2, s = b[0], b[1], b[2], b[3], b[4]
                f.write('{:.1f} {:.1f} {:.1f} {:.1f} {:.3f}\n'.format(x1, y1, (x2 - x1 + 1), (y2 - y1 + 1), s))
                if s>0.4:
                    counts1=counts1+1
                    cv2.rectangle(orig_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    #cv2.putText(orig_image, str(1), (int(x1), int(y1 + 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.putText(orig_image, str(counts1), (80, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            """
            cv2.putText(orig_image, str(counts0), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
            cv2.imwrite(os.path.join(save_path_img+im_dir , (im_name+'.jpg')), orig_image)
            f.close()
            print('event:%d num:%d' % (index + 1, num + 1))

def test_MAFA_img(model_path):
    import glob
    Path = '../data/MIX/MAFA_val/'
    #Path = '/home/yang/datasets/aizoo/val_split/img/'
    Path=Path+'*.jpg'
    #save_path_face= '/home/yang/CenterFace/output/MAFA/MAFA_face/MAFA/'
    #save_path_mask= '/home/yang/CenterFace/output/MAFA/MAFA_mask/MAFA/'
    #os.makedirs(save_path_face)
    #os.makedirs(save_path_mask)
    save_path= '../output/MAFA/MAFA/'
    save_path_img = '../output/MAFA_img/'
    
    if not os.path.exists(save_path):
            os.makedirs(save_path)
            os.makedirs(save_path_img)
    debug = 0          # return the detect result without show
    threshold = 0.05
    TASK = 'ctdet'  
    input_h, intput_w = 512, 512
    opt = opts().init('--task {} --load_model {} --debug {} --vis_thresh {} --input_h {} --input_w {}'.format(
        TASK, MODEL_PATH, debug, threshold, input_h, intput_w).split(' '))
    detector = detector_factory[opt.task](opt)
    Paths = glob.glob(Path)
    
    for i in range(len(Paths)):
        img_path=Paths[i]
        img_name = img_path.split('/')[-1]
        orig_image = cv2.imread(img_path)
        dets = detector.run(img_path)['results']
        f = open(save_path + img_name[0:len(img_name)-4] + '.txt', 'w')
        #f = open(save_path_mask + img_name[0:len(img_name)-4] + '.txt', 'w')
        f.write('{:s}\n'.format('%s' % (img_path)))
        f.write('{:d}\n'.format(len(dets)))
        #ff = open(save_path_face + img_name[0:len(img_name)-4] + '.txt', 'w')
        #ff.write('{:s}\n'.format('%s' % (img_path)))
        #ff.write('{:d}\n'.format(len(dets)))
        counts0 = 0
        counts1 = 0            
        for b in dets[1]:
            x1, y1, x2, y2, s = b[0], b[1], b[2], b[3], b[4]
            #f.write('{:.1f} {:.1f} {:.1f} {:.1f} {:.3f}\n'.format(x1, y1, (x2 - x1 + 1), (y2 - y1 + 1), s))
            if s>0.4:
                counts0=counts0+1
                cv2.rectangle(orig_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                #cv2.putText(orig_image, str(0), (int(x1), int(y1 + 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        for b in dets[2]:
            x1, y1, x2, y2, s = b[0], b[1], b[2], b[3], b[4]
            f.write('{:.1f} {:.1f} {:.1f} {:.1f} {:.3f}\n'.format(x1, y1, (x2 - x1 + 1), (y2 - y1 + 1), s))
            if s>0.4:
                counts1=counts1+1
                cv2.rectangle(orig_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                #cv2.putText(orig_image, str(1), (int(x1), int(y1 + 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(orig_image, str(counts0), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(orig_image, str(counts1), (80, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imwrite(os.path.join(save_path_img+img_name), orig_image)
        print('num:%d' % ( i))
        

def test_AIZOO_img(model_path):
    import glob
    
    Path = '../data/datasets/aizoo/val_split/img/'
    Path=Path+'*.jpg'
    save_path_face= '../output/MAFA/MAFA_face/MAFA/'
    save_path_mask= '../output/MAFA/MAFA_mask/MAFA/'
    os.makedirs(save_path_face)
    os.makedirs(save_path_mask)
    save_path_img = '../output/MAFA_img/'
    
    #os.makedirs(save_path)
    os.makedirs(save_path_img)
    debug = 0          # return the detect result without show
    threshold = 0.05
    TASK = 'ctdet'  
    input_h, intput_w = 512, 512
    opt = opts().init('--task {} --load_model {} --debug {} --vis_thresh {} --input_h {} --input_w {}'.format(
        TASK, MODEL_PATH, debug, threshold, input_h, intput_w).split(' '))
    detector = detector_factory[opt.task](opt)
    Paths = glob.glob(Path)
    
    for i in range(len(Paths)):
        img_path=Paths[i]
        img_name = img_path.split('/')[-1]
        orig_image = cv2.imread(img_path)
        dets = detector.run(img_path)['results']
        f = open(save_path_face + img_name[0:len(img_name)-4] + '.txt', 'w')
        #f = open(save_path_mask + img_name[0:len(img_name)-4] + '.txt', 'w')
        f.write('{:s}\n'.format('%s' % (img_path)))
        f.write('{:d}\n'.format(len(dets)))
        ff = open(save_path_mask + img_name[0:len(img_name)-4] + '.txt', 'w')
        ff.write('{:s}\n'.format('%s' % (img_path)))
        ff.write('{:d}\n'.format(len(dets)))
        counts0 = 0
        counts1 = 0            
        for b in dets[1]:
            x1, y1, x2, y2, s = b[0], b[1], b[2], b[3], b[4]
            f.write('{:.1f} {:.1f} {:.1f} {:.1f} {:.3f}\n'.format(x1, y1, (x2 - x1 + 1), (y2 - y1 + 1), s))
            if s>0.4:
                counts0=counts0+1
                cv2.rectangle(orig_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                #cv2.putText(orig_image, str(0), (int(x1), int(y1 + 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        for b in dets[2]:
            x1, y1, x2, y2, s = b[0], b[1], b[2], b[3], b[4]
            ff.write('{:.1f} {:.1f} {:.1f} {:.1f} {:.3f}\n'.format(x1, y1, (x2 - x1 + 1), (y2 - y1 + 1), s))
            if s>0.4:
                counts1=counts1+1
                cv2.rectangle(orig_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                #cv2.putText(orig_image, str(1), (int(x1), int(y1 + 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(orig_image, str(counts0), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(orig_image, str(counts1), (80, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imwrite(os.path.join(save_path_img+img_name), orig_image)
        print('num:%d' % ( i))
        

if __name__ == '__main__':
    MODEL_PATH = 'D:/CenterMask/exp/ctdet/AIZOO/model_best.pth'
    #test_AIZOO_img(MODEL_PATH)
    # test_wider_Face(MODEL_PATH)
    #test_MAFA_img(MODEL_PATH)
    test_vedio(MODEL_PATH)
