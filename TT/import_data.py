import pymysql


def get_data():


    conn = pymysql.connect(host='127.0.0.1',port=3306, user='root', passwd='123', db='rj_mdt',charset='utf8')

    cursor = conn.cursor()
    
    
    cursor.execute("SELECT * FROM stats;")
    data=cursor.fetchall()
    

    conn.commit()
    cursor.close()
    conn.close()
    
    
    return data

def get_data_D():
    conn = pymysql.connect(host='127.0.0.1', 
    port=3306, user='root', passwd='123', db='rj_mdt',
     charset='utf8')

    cursor = conn.cursor()
    
    
    cursor.execute("SELECT stats.id,stats.vector,cases.side,votes.user_id,votes.pre_targeted_therapy_id,votes.targeted_therapy_id FROM rj_mdt.stats  left join rj_mdt.cases  on stats.patient_id = cases .patient_id left join rj_mdt.votes on cases.id=votes.case_id where votes.targeted_therapy_id is not NULL and votes.pre_targeted_therapy_id is not NULL")
    data=cursor.fetchall()

    conn.commit()
    cursor.close()
    conn.close()
    
    
    return data