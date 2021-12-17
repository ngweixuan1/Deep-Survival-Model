WITH ischemic_patients AS (
    SELECT DISTINCT dia.SUBJECT_ID, ite.LABEL as features, ite.ITEMID, cha.VALUENUM, DATE(cha.CHARTTIME) as record_date, DATE(pat.DOB) as date_of_birth
    FROM `physionet-data.mimiciii_clinical.diagnoses_icd` as dia
    LEFT JOIN `physionet-data.mimiciii_clinical.chartevents` as cha
        ON cha.SUBJECT_ID = dia.SUBJECT_ID
    LEFT JOIN `physionet-data.mimiciii_clinical.patients` as pat
        ON pat.SUBJECT_ID = dia.SUBJECT_ID
    LEFT JOIN `physionet-data.mimiciii_clinical.d_items` as ite
        ON cha.ITEMID = ite.ITEMID
    WHERE (ICD9_CODE LIKE '410%' 
            OR ICD9_CODE LIKE '411%' 
            OR ICD9_CODE LIKE '413%'
            OR ICD9_CODE LIKE '414%'
            OR ICD9_CODE = '412')
    AND ite.ITEMID IN (220603,22062,225693,220050,220051,227688,227007,22744,227429,227444,220277,220283, 220046,220047,
                        227468,220621,226537,5166,225671,227445,227463,220615,220045,220228,3684,3685,6509,6510,220048,
                        220052, 220056, 220058) 
)
SELECT
    subject_id
    , record_date
    , features AS feature_name
    , ROUND(AVG(value), 3) AS value
FROM
( 
SELECT
    ie.subject_id
    , ip.record_date
    , ip.features 
    , ip.VALUENUM AS value
    -- , CASE 
    --     WHEN ad.deathtime IS NULL THEN 0
    --     WHEN DATE(ad.deathtime) < ip.record_date THEN 0
    --     ELSE 1
    --     END AS Death_Flag
    -- , DATETIME_DIFF(ip.record_date, ip.date_of_birth, YEAR) as patient_age
FROM `physionet-data.mimiciii_clinical.icustays` ie
INNER JOIN ischemic_patients as ip 
    ON ip.SUBJECT_ID = ie.SUBJECT_ID
INNER JOIN `physionet-data.mimiciii_clinical.admissions` ad
    ON ie.SUBJECT_ID = ad.SUBJECT_ID
WHERE ip.valuenum IS NOT NULL 
ORDER BY ie.subject_id, ip.record_date
)
GROUP BY subject_id, record_date, features
ORDER BY subject_id, record_date