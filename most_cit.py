from dask.multiprocessing import get
# from tqdm.auto import tqdm 
from tqdm import tqdm

tqdm.pandas()



import pandas as pd 
from pathlib import Path
from tqdm.auto import tqdm
import time
import textwrap 
from rich import print
import pyarrow.parquet as pypq


if pd.__version__ >= '2.1':
  print('Using PyArrow strings!')
  pd.options.future.infer_string = True


top_cit_pd = pd.read_csv('works_br_ascending_cit_ge30')
top4000 = top_cit_pd #.head(40)


def peek_parquet(path):
    """
    peeks at a parquet file (or a directory containing parquet files) and prints the following:
    * Path
    * schema
    * number of pieces (fragments)
    * number of rows 
    """
    if isinstance(path, str):
        path = Path(path)
        
    parq_file = pypq.ParquetDataset(path)
    piece_count = len(parq_file.fragments)
    schema = textwrap.indent(parq_file.schema.to_string(), ' '*4)
    row_count = sum(frag.count_rows() for frag in parq_file.fragments)
    
    st = [
        f'Name: {path.stem!r}',  
        f'Path: {str(path)!r}',
        f'Files: {piece_count:,}',
        f'Rows: {row_count:,}',
        f'Schema:\n{schema}',
        f'5 random rows:',
    ]
    print('\n'.join(st))
    sample_df = parq_file.fragments[0].head(5).to_pandas()  # read 5 rows from the first fragment
    print(sample_df)

    return

def read_parquet(path, engine='pyarrow', convert_dtypes=True, **args):
    """
    Read a parquet file
    """
    if isinstance(path, str):
        path = Path(path)

    if 'filters' in args:
        engine = 'pyarrow'
    
    if not path.name.endswith('.parquet'):
        ## check if a file exists without the extension 
        dir_exists = path.exists()
        if not dir_exists:  # try adding the parquet extension
            if path.with_suffix('.parquet').exists():
                path = path.with_suffix('.parquet')
                
    name = path.stem
        
    print(f'\nReading {name!r} from {str(path)!r} using {engine=}')

    tic = time.time()
    df = pd.read_parquet(path, engine=engine, **args)
    toc = time.time()
    print(f'Read {len(df):,} rows from {path.stem!r} in {toc-tic:.2f} sec.')
    
    if convert_dtypes:
        tic = time.time()
        size_before = df.memory_usage(deep=True).sum() / 1024 / 1024 / 1024

        string_cols_d = {}
        # string_type = 'string[python]'
        string_type = 'string[pyarrow]'
        for col, dtype in df.dtypes.to_dict().items():
            if pd.__version__ <= '2.1':
              if dtype == 'object' or dtype == 'string':  # convert object columns to string
                  string_cols_d[col] = string_type
            if col == 'type' or col == 'concept_name':
                if dtype != 'category':
                    string_cols_d[col] = 'category'
            if col == 'publication_month':
                if dtype != 'uint8':
                    string_cols_d[col] = 'uint8'
            if col == 'score':
                if dtype != 'float16':
                    string_cols_d[col] = 'float16'
                
        df = df.astype(string_cols_d) 
        
        size_after = df.memory_usage(deep=True).sum() / 1024 / 1024 / 1024
        toc = time.time()
        print(f'Converting dtypes took {toc-tic:.2f} sec. Size before: {size_before:.2f}GB, after: {size_after:.2f}GB')
    print(df.head(3))
    return df


parq_basepath = Path('/N/project/openalex/slices/subset-1990-2022/dec-2023/')

## get the work ids for the authors 
focal_papers_idxs = set(
    top4000['work_id'].values
)

focal_cits = read_parquet(
    parq_basepath / 'works_referenced_works',
    engine='pyarrow',
    columns=['work_id', 'referenced_work_id'],
    filters=[[('referenced_work_id', 'in', focal_papers_idxs)]],
)

focal_cits_idxs = set(
    focal_cits.work_id
)

potential_cits_of_cits = read_parquet(
    parq_basepath / 'works_referenced_works',
    engine='pyarrow',
    columns=['work_id', 'referenced_work_id'],
    filters=[[('referenced_work_id', 'in', focal_cits_idxs)]],
)
print(f'{len(focal_cits_idxs)=:,}')

works_br_year = read_parquet(
    parq_basepath / 'works',
    engine='pyarrow',
    columns=['work_id', 'publication_year'],
    filters=[[('work_id', 'in', focal_papers_idxs | focal_cits_idxs)]],
)

refs = read_parquet(
    parq_basepath / 'works_referenced_works',
    engine='pyarrow',
    columns=['work_id', 'referenced_work_id'],
    filters=[[('work_id', 'in', focal_papers_idxs)]],
)

def get_disruption_idx1(focal_cits, focal_refs):
    idx_numerator = 0
    test = potential_cits_of_cits[potential_cits_of_cits['work_id'].isin(focal_cits)]
    for paper, paper_refs in test.groupby('work_id'):
        a1 = set(paper_refs['referenced_work_id'].values)
        intersection = a1 & set(focal_refs['referenced_work_id'].values)
        f_it = 1
        if len(intersection) > 0:
            b_it = 1
        else:
            b_it = 0
    
        idx_numerator += -2*f_it*b_it + f_it    
    
    
    idx_denominator = len(focal_cits)
    disruption_idx = idx_numerator/idx_denominator
    return disruption_idx

def bla(focal_paper_cits):
    #global refs
    paper_id = focal_paper_cits['referenced_work_id'].iloc[0]

    focal_paper_cits = focal_paper_cits.merge(works_br_year, how='left',left_on='work_id',right_on='work_id')
    year = works_br_year.loc[paper_id]['publication_year']
    total_cits = len(focal_paper_cits)
    focal_paper_cits = focal_paper_cits[focal_paper_cits['publication_year'] - year <= 5]
    focal_paper_cits = set(focal_paper_cits['work_id'])
    focal_refs = refs[refs['work_id'] == paper_id]
    
    if len(focal_refs) == 0:
        return None, None #, None, None, None
    if len(focal_paper_cits) < 30:
        return None, None #, None, None, None
    disr_idx = get_disruption_idx1(focal_paper_cits, focal_refs)
    return paper_id, disr_idx #, len(focal_refs), total_cits, len(focal_paper_cits)

# paper_disruption = []
i = 0

# ainda nao tem janela de tempo de 5 anos depois do artigo publicado
import swifter

paper_disruption = focal_cits.swifter.groupby('referenced_work_id').apply(bla)
paper_disruption.to_csv('paper_di_18jul.csv')