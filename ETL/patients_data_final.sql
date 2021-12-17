WITH admission_time AS
(
  SELECT
      p.subject_id, p.dob, p.gender
      , MIN (a.admittime) AS first_admittime
  FROM `physionet-data.mimiciii_clinical.patients` p
  INNER JOIN `physionet-data.mimiciii_clinical.admissions` a
  ON p.subject_id = a.subject_id
  GROUP BY p.subject_id, p.dob, p.gender
  ORDER BY p.subject_id
)
, ischemic_patients AS (
    SELECT DISTINCT dia.SUBJECT_ID, dia.HADM_ID
    FROM `physionet-data.mimiciii_clinical.diagnoses_icd` as dia
    LEFT JOIN `physionet-data.mimiciii_clinical.chartevents` as cha
        ON cha.SUBJECT_ID = dia.SUBJECT_ID
    LEFT JOIN `physionet-data.mimiciii_clinical.d_items` as ite
        ON cha.ITEMID = ite.ITEMID
    WHERE (ICD9_CODE LIKE '410%' 
            OR ICD9_CODE LIKE '411%' 
            OR ICD9_CODE LIKE '413%'
            OR ICD9_CODE LIKE '414%'
            OR ICD9_CODE LIKE '412%')
    AND ite.ITEMID IN (220603,22062,225693,220050,220051,227688,227007,22744,227429,227444,220277,220283, 220046,220047,
                        227468,220621,226537,5166,225671,227445,227463,220615,220045,220228,3684,3685,6509,6510,220048,
                        220052, 220056, 220058) 
)
, serv AS (
    SELECT
        icu.hadm_id,
        icu.icustay_id,
        se.curr_service,
        IF(curr_service like '%SURG' OR curr_service = 'ORTHO', 1, 0) AS surgical,
        RANK() OVER (PARTITION BY icu.hadm_id ORDER BY se.transfertime DESC) as rank
    FROM `physionet-data.mimiciii_clinical.icustays` AS icu
    LEFT JOIN `physionet-data.mimiciii_clinical.services` AS se
    ON icu.hadm_id = se.hadm_id
    AND se.transfertime < DATETIME_ADD(icu.intime, INTERVAL 12 HOUR)
)

SELECT * except (stay_num, death_in_hospital, death_in_icu), 
    CASE WHEN (death_in_hospital + death_in_icu) < 1 THEN 0 ELSE 1 END AS death_flag
FROM
(
    SELECT 
        isp.subject_id, 
        ie.hadm_id AS admission_id, 
        DATE(admt.first_admittime) as first_admittime,
        -- ie.icustay_id,
        ROW_NUMBER() over (partition by ie.subject_id order by adm.admittime DESC, ie.intime DESC) as stay_num,
        --ie.intime AS begin_time_icu, 
        DATE(ie.outtime) AS end_time_icu,
        DATETIME_DIFF(ie.intime, admt.dob, YEAR) as patient_age_admission, 
        -- patient death in hospital is stored in the admissions table
        DATE(adm.deathtime) as deathtime,
        admt.gender,
        se.curr_service,
        se.surgical,
        -- adm.dischtime AS discharge_time,
        adm.MARITAL_STATUS as marital_status, 
        DATETIME_DIFF(ie.intime, adm.admittime, DAY) as num_days_icu,
        CASE
            WHEN DATETIME_DIFF(adm.admittime, pat.dob, YEAR) <= 1
                THEN 'neonate'
            WHEN DATETIME_DIFF(adm.admittime, pat.dob, YEAR) <= 14
                THEN 'middle'
            WHEN DATETIME_DIFF(adm.admittime, pat.dob, YEAR) > 89
                THEN '>89'
            ELSE 'adult'
        END AS age_group,
        -- the "hospital_expire_flag" field in the admissions table indicates if a patient died in-hospital
        CASE
            WHEN adm.hospital_expire_flag = 1 then 1
        ELSE 0
        END AS death_in_hospital,
        -- note also that hospital_expire_flag is equivalent to "Is adm.deathtime not null?"
        CASE
            WHEN adm.deathtime BETWEEN ie.intime and ie.outtime
                THEN 1
            -- sometimes there are typographical errors in the death date, so check before intime
            WHEN adm.deathtime <= ie.intime
                THEN 1
            WHEN adm.dischtime <= ie.outtime
                AND adm.discharge_location = 'DEAD/EXPIRED'
                THEN 1
            ELSE 0
            END AS death_in_icu,
        DATE(pat.dob) as date_of_birth
    FROM `physionet-data.mimiciii_clinical.icustays` ie
    INNER JOIN ischemic_patients AS isp 
        ON ie.SUBJECT_ID = isp.SUBJECT_ID
    INNER JOIN `physionet-data.mimiciii_clinical.patients` pat
    ON ie.subject_id = pat.subject_id
    INNER JOIN `physionet-data.mimiciii_clinical.admissions` adm
    ON ie.hadm_id = adm.hadm_id
    INNER JOIN admission_time AS admt
    ON ie.SUBJECT_ID = admt.SUBJECT_ID
    INNER JOIN serv AS se
        ON ie.HADM_ID = se.hadm_id
        AND ie.ICUSTAY_ID = se.icustay_id
    WHERE DATETIME_DIFF(adm.admittime, pat.dob, YEAR) >= 15 AND DATETIME_DIFF(adm.admittime, pat.dob, YEAR) < 90
    ORDER BY ie.SUBJECT_ID
)
WHERE stay_num = 1