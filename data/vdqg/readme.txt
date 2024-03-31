Visual Discriminative Question Generation (VDQG) dataset
version: 1.0


1. Files
	- vdqg.py				: wrapper class "VDQG" and evaluation tools.
	
	- annotations/
		- annotation.json	: annotation data file.
		- human_result.json	: human written questions for evaluation of human performance.


2. VDQG Class

	Please refer to vdqg.py for details.

	2.1 Attributes

	-version: Version of dataset
	-annotation: (dict) all annotations. Each key-value tuple (s_id, anno) is a sample's annotation, where
	    -s_id: (str) unique sample id
	    -anno: (dict) annotation containing:
	        -id: (str) s_id
	        -object: (list) object list (2 objects)
	            -VG_image_id: (str) visual genome image id
	            -VG_object_id: (str) visual genome object id
	            -bbox: (list) object bounding box [x1, y1, x2, y2]
	        -question: (list of str) question annotations
	        -question_label: (list of int) question labels, where "-1" means negative, "1" means weak positive, and "2" means strong positive
	-hard_set_ids: (list of str) the id list of hard samples

	
	2.2 Methods

	- load_human_result(...)
		load human written questions

	- eval_delta_bleu(...)
		evaluate generated questions using DeltaBLEU metric

	- eval_coco(...)
		evaluate generated questions using coco-caption metrics


3. Requirements
	
	The evaluation tool requires nltk package and coco-caption api.


4. Paper
	
	Learning to Disambiguate by Asking Discriminative Questions (ICCV 2017)



For more information please contact ly015@ie.cuhk.edu.hk
	