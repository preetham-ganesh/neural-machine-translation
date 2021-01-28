import tensorflow_datasets as tfds
import os
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)

def convert_tensor_to_lines(tensor, name):
	lines_en = []
	lines_de = []
	c = 1
	for i in tensor:
		try:
			x = str(i['en'].decode("utf-8"))
			y = str(i['de'].decode("utf-8"))
			lines_en.append(x)
			lines_de.append(y)
			if len(lines_en) == 5000000 and len(lines_de) == 5000000:
				en_text = lines_to_text(lines_en, '\n')
				dataset_save(en_text, name+'_'+str(c)+'.en')
				del en_text
				de_text = lines_to_text(lines_de, '\n')
				dataset_save(de_text, name+'_'+str(c)+'.de')
				del de_text
				lines_en, lines_de = [], []
				print(str(c*5000000)+' rows saved')
				print()
				c += 1
		except:
			continue
	en_text = lines_to_text(lines_en, '\n')
	dataset_save(en_text, name+'_'+str(c)+'.en')
	del en_text
	de_text = lines_to_text(lines_de, '\n')
	dataset_save(de_text, name+'_'+str(c)+'.de')
	del de_text

def lines_to_text(lines, sep):
    text = ''
    for i in range(len(lines)):
        if i == len(lines) - 1:
            text += str(lines[i])
        else:
            text += str(lines[i]) + sep
    return text

def dataset_save(text, name):
    f = open('/home/preetham/Documents/language-translation/nmt-de-en/data/original/dataset_10/'+name, 'w', encoding='utf-8')
    f.write(text)
    f.close()

def main():
	train = tfds.as_numpy(tfds.load('wmt19_translate/de-en', split='train', shuffle_files=True))
	val = tfds.as_numpy(tfds.load('wmt19_translate/de-en', split='validation', shuffle_files=True))
	print()
	print('Datasets loaded successfully')
	print()
	convert_tensor_to_lines(train, 'train')
	del train
	convert_tensor_to_lines(val, 'val')
	del val
	print('Datasets converted to lines')
	print()
	
main()
