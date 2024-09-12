SELECT 
mediacontent_pagetitle_program as naam,
brand_contentbrand as merk,
mediacontent_media_content_type as type
FROM (
SELECT 
mediacontent_pagetitle_program,
brand_contentbrand,
mediacontent_media_content_type
FROM derived_prod.vrtmax_catalog_mediaid_history
WHERE year = 2024 and month = 9 and day = 12 and hour = 9 and
offering_publication_planneduntil > current_timestamp
and mediacontent_pagetitle_program is not null 
and brand_contentbrand is not null and mediacontent_media_content_type is not null)
GROUP BY mediacontent_pagetitle_program,brand_contentbrand,mediacontent_media_content_type