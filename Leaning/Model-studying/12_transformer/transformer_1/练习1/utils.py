import tensorflow as tf
import json
import os,re
import logging

logging.basicConfig(level=logging.INFO)

def calc_num_batches(total_num,batch_size):
    return total_num // batch_size + int(total_num % batch_size != 0 )

def convert_idx_to_token_tensor(inputs,idx2token):
    return tf.py_func(lambda ipt:' '.join(idx2token[elem] for elem in inputs), [inputs],tf.string)

def postprocess(hypotheses,idx2token):
    _hypotheses = []
    for h in hypotheses:
        sent = ''.join(idx2token[idx] for idx in h)
        sent = sent.split('</s>')[0].strip()    # todo 这里的/s是不是应该是/S
        sent = sent.replace('__',' ')
        _hypotheses.append(sent.strip())
    return _hypotheses

def save_hparams(hparams,path):
    if not os.path.exists(path):
        os.mkdir(path)
    hp = json.dumps(vars(hparams))
    with open(os.path.join(path,'hparams'),'w') as fout:
        fout.write(hp)

def load_hparams(parser,path):
    if not os.path.isdir(path):
        path = os.path.dirname(path)
    d = open(os.path.join(path,'hparams'),'r').read()
    flag2val = json.loads(d)
    for f, v in flag2val.items():
        parser.f = v

def save_variable_speces(path):
    def _get_size(shp):
        size = 1
        for d in range(len(shp)):
            size *= int(shp[d])
        return size

    params, num_params = [], 0
    for v in tf.global_variables():
        params.append('{}==={}'.format(v.name,v.shape))
        num_params += _get_size(v.shape)
    print(('num_params: ',num_params))
    with open(path ,'w') as fout:
        fout.write('num_params: {}\n'.format(num_params))
        fout.write('\n'.join(params))
    logging.info('Variables info has been saved.')

def get_hypotheses(num_batches,num_samples,sess,tensor,dict):
    hypotheses = []
    for _ in range(num_batches):
        h = sess.run(tensor)
        hypotheses.extend(h.tolist())
    hypotheses = postprocess(hypotheses,dict)

    return hypotheses[:num_samples]

def calc_bleu(ref,translation):
    get_bleu_score = 'perl multi-bleu.prel {} < {} > {}'.format(ref,translation,'temp')
    os.system(get_bleu_score)
    bleu_score_report = open('temp','r').read()
    with open(translation,'a') as fout:
        fout.write('\n{}'.format(bleu_score_report))
    try:
        score = re.findall('BLEU = ([^,]+)',bleu_score_report)
        new_translation = translation + 'B{}'.format(score)
        os.system('mv {} {}'.format(translation,new_translation))
        os.remove(translation)
    except:
        pass
    os.remove('temp')

def create_model_and_embedding(session,Model_class,path,config,is_train):
    model = Model_class(config,is_train)
    ckpt = tf.train.get_checkpoint_state(path)
    if ckpt and tf.train.checkpoint_exists(session,ckpt.model_checkpoint_path):
        model.saver.restor(session,ckpt.model_checkpoint_path)
    else:
        session.run(tf.global_variables_initializer())
    return model

def save_model(sess,model,path,logger):
    checkpoint_path = os.path.join(path,'chatbot.ckpt')
    model.saver.save(sess,checkpoint_path)
    logger.info('model saved')
