# scancel from the list below

list=(
    "2384204"
    "2384206"
    "2384207"
    "2384208"
    "2384209"
    "2384210"
    "2384211"
    "2384213"
    "2384215"
    "2384216"
    "2384217"
    "2384218"
    "2384220"
    "2384221"
    "2384222"
    "2384223"
    "2384226"
    "2384228"
    "2384230"
    "2384231"
    "2384233"
    "2384234"
    "2384264"
    "2384262"
    "2384261"
    "2384259"
    "2384257"
    "2384255"
    "2384253"
    "2384251"
    "2384249"
    "2384244"
    "2384242"
    "2384240"
    "2384238"
    "2384236"
)

for i in "${list[@]}"
do
    scancel $i
done
