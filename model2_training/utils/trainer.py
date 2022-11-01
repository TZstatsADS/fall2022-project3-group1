import time
import numpy as np
import tensorflow as tf
from datetime import datetime

"""
Author: Chengming He
A trainer for the network
"""

def trainer(model,batch_size,input_size,N_train,N_val,train_data,validation_data,epochs,optimizer,loss_fn,train_acc_metric,val_acc_metric,name,verbose=1,save=False):
    history = [[],[],[]]
    time_0 = time.time()
    pre_val_acc = 0
    for epoch in range(epochs):
        if verbose:
            print("Epoch %d/%d" % (epoch+1,epochs))        
        start_time = time.time()
        step = 0
        for X_train,y_clean_batch_train in train_data:

            if input_size == 2:
                x_batch_train, y_noisy_batch_train = X_train
            if input_size == 1:
                x_batch_train = X_train
            with tf.GradientTape() as tape:
                if input_size ==2:
                    logits = model([x_batch_train,y_noisy_batch_train])
                if input_size == 1:
                    logits = model([x_batch_train])
                loss_value = loss_fn(y_clean_batch_train,logits)
            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            train_acc_metric.update_state(y_clean_batch_train, logits)
            if verbose == 2:
                if step % 20 == 0:
                    print(
                        "Training loss at step %d: %.4f"
                        % (step, float(loss_value))
                    )
            step += 1
            if step > N_train/batch_size:
                break

        history[0].append(loss_value)
        train_acc = train_acc_metric.result()
        train_acc_metric.reset_states()
        
        step = 0
        for X_val,y_clean_batch_val in validation_data:
            if input_size == 2:
                x_batch_val, y_noisy_batch_val = X_val
                val_logits = model([x_batch_val, y_noisy_batch_val])
            if input_size == 1:
                x_batch_val = X_val
                val_logits = model([x_batch_val])
            
            val_acc_metric.update_state(y_clean_batch_val, val_logits)
            step += 1
            if step > N_val/batch_size:
                break
        val_acc = val_acc_metric.result()
        if pre_val_acc < val_acc:
            if save:
                model.save_weights("models/{}_epoch_{}_{}.h5".format(name,epoch+1,str(datetime.now())))
            pre_val_acc = val_acc
        val_acc_metric.reset_states()
        history[1].append(train_acc)
        history[2].append(val_acc)
        if verbose:
            print("Training accuracy: %.4f" % (float(train_acc),)
                  ,"Validation accuracy: %.4f" % (float(val_acc),),"Time taken: %.2fs" % (time.time() - start_time))
    total_time=time.time()-time_0
    if verbose:
        print("Total training time: ",total_time, "s")

    return np.array(history)
    
