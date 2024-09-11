WITH dataset as (SELECT
mediacontent_pageid_duplicate,
transform(mediacontent_episode_castlist, r -> concat_ws( ' ',r.firstname,r.surname)) as mediacontent_episode_castlist
FROM (
SELECT 
mediacontent_pageid as mediacontent_pageid_duplicate,mediacontent_episode_castlist,
row_number() over (partition by mediacontent_pageid ORDER BY mediacontent_page_modificationdate desc) as rn
FROM marketing_prod.aem_vrtvideo_datariver
WHERE year = 2024  and mediacontent_episode_castlist is not null )
WHERE rn = 1)

select distinct persoon
from dataset,
unnest(mediacontent_episode_castlist) as t(persoon)