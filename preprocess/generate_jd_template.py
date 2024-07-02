from typing import Optional
from langchain_core.pydantic_v1 import BaseModel, Field
from pydantic import BaseModel, Field, HttpUrl, EmailStr
from typing import List, Optional
from sentence_transformers import SentenceTransformer
import json
from langchain_openai import OpenAIEmbeddings
import pandas as pd
import dotenv
from typing import Optional
import csv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser




class JobDescription(BaseModel):
    title: Optional[str] = Field(None, description="职位名称", example="软件工程师")
    company: Optional[str] = Field(None, description="公司名称", example="OpenAI")
    location: Optional[str] = Field(None, description="工作地点", example="加利福尼亚州旧金山")
    job_type: Optional[str] = Field(None, description="工作类型", example="全职")
    salary_range: Optional[str] = Field(None, description="薪资范围", example="¥200,000 - ¥300,000")
    responsibilities: Optional[List[str]] = Field(None, description="工作职责列表", example=["开发软件", "与团队合作"])
    qualifications: Optional[List[str]] = Field(None, description="资格要求列表", example=["计算机科学学士学位", "3年以上软件开发经验"])
    benefits: Optional[List[str]] = Field(None, description="福利列表", example=["健康保险", "退休计划"])
    
def generate_jd_template(jd_path, save_template_path):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert extraction algorithm. "
                "Only extract relevant information from the text. "
                "If you do not know the value of an attribute asked to extract, "
                "return null for the attribute's value.",
            ),
            ("human", "{text}"),
        ]
    )
    # llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
    llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0)
    runnable = prompt | llm.with_structured_output(schema=JobDescription) 

    # embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    embeddings_model = SentenceTransformer(
        "all-MiniLM-L6-v2", device="cuda"
    )

    df = pd.read_csv(jd_path)
    # print(len(df)) # 4856
    column_names = ['uuid', 'JD_content', 'template', 'embedding']
    with open(save_template_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(column_names)
        for idx, row in df.iterrows():
            # if idx != 879:
            #     continue
            JD_content = row['JD_content']

            try:
                response = runnable.invoke({"text": JD_content})
            except Exception as e:
                print(e)
                print(f"write {idx}-th jd to {save_template_path} fail!")
                continue

            data_json = json.dumps(dict(response), ensure_ascii=False, indent=2)
            text = ""
            for key, value in dict(response).items():
                if value != None and value != 'null':
                    text += str(value) + " "
            # embedding = embeddings.embed_documents([text])[0]

            embedding = embeddings_model.encode(
                [text],
                show_progress_bar=True,
            )[0]

            new_row = []
            new_row.append(row['uuid'])
            new_row.append(row['JD_content'])
            new_row.append(data_json)  # template
            new_row.append(embedding)
            writer.writerow(new_row)
            print(f"write {idx}-th jd to {save_template_path} success!")
            # if idx == 3:
            #     break

    print("write jd template success!")
            


if __name__ == '__main__':

    jd_path = '/autodl-fs/data/wang/falcon/csvfile/JD.csv'
    save_template_path = '/autodl-fs/data/song/jd_cv_gpt4_template/jd_template_embedding.csv'
    # save_template_path = 'jd_template_embedding.csv'

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert extraction algorithm. "
                "Only extract relevant information from the text. "
                "If you do not know the value of an attribute asked to extract, "
                "return null for the attribute's value.",
            ),
            ("human", "{text}"),
        ]
    )
    # llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
    llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0)
    runnable = prompt | llm.with_structured_output(schema=JobDescription) 

    # embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    embeddings_model = SentenceTransformer(
        "all-MiniLM-L6-v2", device="cuda"
    )

    df = pd.read_csv(jd_path)
    # print(len(df)) # 4856
    column_names = ['uuid', 'JD_content', 'template', 'embedding']
    with open(save_template_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(column_names)
        for idx, row in df.iterrows():
            # if idx != 879:
            #     continue
            JD_content = row['JD_content']

            try:
                response = runnable.invoke({"text": JD_content})
            except Exception as e:
                print(e)
                print(f"write {idx}-th jd to {save_template_path} fail!")
                continue

            data_json = json.dumps(dict(response), ensure_ascii=False, indent=2)
            text = ""
            for key, value in dict(response).items():
                if value != None and value != 'null':
                    text += str(value) + " "
            # embedding = embeddings.embed_documents([text])[0]

            embedding = embeddings_model.encode(
                [text],
                show_progress_bar=True,
            )[0]

            new_row = []
            new_row.append(row['uuid'])
            new_row.append(row['JD_content'])
            new_row.append(data_json)  # template
            new_row.append(embedding)
            writer.writerow(new_row)
            print(f"write {idx}-th jd to {save_template_path} success!")
            # if idx == 3:
            #     break

    print("write jd template success!")
            

