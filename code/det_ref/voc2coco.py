"""voc2coco.py — convert VOC2007 test to a COCO-format dir reusable by pilot1_det_coverage.py.
Produces <out>/val2017/*.jpg (symlinks) + <out>/annotations/instances_val2017.json with COCO
category_ids (VOC's 20 classes mapped to their COCO names), so the COCO-trained detector evaluates
cross-dataset with no code change.
Run on server: python3 voc2coco.py --voc ~/data/VOCdevkit/VOC2007 --out ~/data/voc_as_coco
"""
import os, json, argparse, xml.etree.ElementTree as ET

VOC2COCO_NAME = {  # VOC name -> COCO name
    "aeroplane":"airplane","bicycle":"bicycle","bird":"bird","boat":"boat","bottle":"bottle",
    "bus":"bus","car":"car","cat":"cat","chair":"chair","cow":"cow","diningtable":"dining table",
    "dog":"dog","horse":"horse","motorbike":"motorcycle","person":"person","pottedplant":"potted plant",
    "sheep":"sheep","sofa":"couch","train":"train","tvmonitor":"tv"}
# COCO 80-class name -> id (standard)
COCO_NAME2ID = {n:i for n,i in [
 ("person",1),("bicycle",2),("car",3),("motorcycle",4),("airplane",5),("bus",6),("train",7),
 ("boat",9),("bird",16),("cat",17),("dog",18),("horse",19),("sheep",20),("cow",21),
 ("bottle",44),("chair",62),("couch",63),("potted plant",64),("dining table",67),("tv",72)]}


def main(a):
    img_dir = os.path.join(a.voc, "JPEGImages")
    ann_dir = os.path.join(a.voc, "Annotations")
    ids = [l.strip() for l in open(os.path.join(a.voc, "ImageSets/Main/test.txt"))]
    os.makedirs(os.path.join(a.out, "val2017"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "annotations"), exist_ok=True)
    cats = [{"id": COCO_NAME2ID[n], "name": n} for n in sorted(set(VOC2COCO_NAME.values()))]
    images, anns = [], []
    aid = 1
    for k, iid in enumerate(ids):
        xml = os.path.join(ann_dir, iid + ".xml")
        if not os.path.exists(xml):
            continue
        r = ET.parse(xml).getroot()
        W = int(r.find("size/width").text); H = int(r.find("size/height").text)
        imgid = int(iid.replace("_", ""))   # VOC ids like 000001 -> int
        src = os.path.join(img_dir, iid + ".jpg"); dst = os.path.join(a.out, "val2017", f"{imgid:012d}.jpg")
        if not os.path.exists(dst):
            try: os.symlink(src, dst)
            except OSError: import shutil; shutil.copy(src, dst)
        images.append({"id": imgid, "file_name": f"{imgid:012d}.jpg", "width": W, "height": H})
        for obj in r.findall("object"):
            nm = obj.find("name").text
            if nm not in VOC2COCO_NAME:
                continue
            if int(obj.find("difficult").text or 0) == 1:
                continue
            b = obj.find("bndbox")
            x1, y1, x2, y2 = (float(b.find(t).text) for t in ("xmin","ymin","xmax","ymax"))
            w, h = x2 - x1, y2 - y1
            anns.append({"id": aid, "image_id": imgid, "category_id": COCO_NAME2ID[VOC2COCO_NAME[nm]],
                         "bbox": [x1, y1, w, h], "area": w * h, "iscrowd": 0})
            aid += 1
    json.dump({"images": images, "annotations": anns, "categories": cats},
              open(os.path.join(a.out, "annotations", "instances_val2017.json"), "w"))
    print(f"VOC->COCO: {len(images)} images, {len(anns)} anns -> {a.out}")


if __name__ == "__main__":
    P = argparse.ArgumentParser()
    P.add_argument("--voc", default=os.path.expanduser("~/data/VOCdevkit/VOC2007"))
    P.add_argument("--out", default=os.path.expanduser("~/data/voc_as_coco"))
    main(P.parse_args())
