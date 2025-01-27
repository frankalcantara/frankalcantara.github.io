import sqlite3
from datetime import datetime

def adapt_datetime(dt):
   return dt.isoformat()

def init_gan_db():
   sqlite3.register_adapter(datetime, adapt_datetime)
   conn = sqlite3.connect('gan_config.db')
   cursor = conn.cursor()
   
   cursor.execute('''
   CREATE TABLE IF NOT EXISTS gan_configs (
       id INTEGER PRIMARY KEY AUTOINCREMENT,
       user_id INTEGER NOT NULL,
       modified_at TEXT NOT NULL,
       latent_dim INTEGER NOT NULL,
       temperature REAL NOT NULL,
       dropout_rate REAL NOT NULL, 
       learning_rate REAL NOT NULL,
       adam_beta1 REAL NOT NULL,
       adam_beta2 REAL NOT NULL,
       num_epochs INTEGER NOT NULL,
       batch_size INTEGER NOT NULL,
       test_size INTEGER NOT NULL,
       num_samples INTEGER NOT NULL,
       distance_metric TEXT NOT NULL,
       gen_layer1 INTEGER NOT NULL,
       gen_layer2 INTEGER NOT NULL,
       gen_layer3 INTEGER NOT NULL,
       disc_layer1 INTEGER NOT NULL,
       disc_layer2 INTEGER NOT NULL,
       disc_layer3 INTEGER NOT NULL,
       training_seed INTEGER NOT NULL
   )''')

   cursor.execute('''
   INSERT OR IGNORE INTO gan_configs (
       user_id, modified_at, latent_dim, temperature, dropout_rate,
       learning_rate, adam_beta1, adam_beta2, num_epochs, batch_size,
       test_size, num_samples, distance_metric, gen_layer1, gen_layer2,
       gen_layer3, disc_layer1, disc_layer2, disc_layer3, training_seed
   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
   ''', (
       1,
       datetime.now(),
       50,     # latent_dim 
       0.5,    # temperature
       0.3,    # dropout_rate
       0.0002, # learning_rate
       0.5,    # adam_beta1
       0.999,  # adam_beta2
       150,    # num_epochs
       32,     # batch_size
       100,    # test_size
       50000,  # num_samples
       'manhattan', # distance_metric
       128,    # gen_layer1
       256,    # gen_layer2
       128,    # gen_layer3
       128,    # disc_layer1
       256,    # disc_layer2
       128,    # disc_layer3
       int(time.time_ns()) % (2**32)  # training_seed
   ))

   conn.commit()
   conn.close()

if __name__ == "__main__":
   init_gan_db()