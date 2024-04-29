from py2neo import Graph,Node,Relationship,NodeMatcher
import pandas as pd

from geopy.distance import geodesic  #给定经纬度，计算两个地理位置的距离
property_path='property.csv'
house_path='house.csv'
street_path='street.csv'
region = 'region.csv'

df_property=pd.read_csv(property_path)
df_house = pd.read_csv(house_path)
df_street=pd.read_csv(street_path)
df_region=pd.read_csv(region)

graph = Graph("http://localhost:7474", auth=("neo4j", "thywdy123"),name="neo4j")
from tqdm import tqdm
#构造实体
def create_property_node(data):
    property= Node("Property",
                 id=int(data['id']),
                 house=int(data['house_id'])
                 )
    return property
def create_house_node(data):
    house = Node("House",
                    house=int(data['house_id']),
                    street=int(data['street_id']),
                    lat=float(data['geo_lat']),
                    lon=float(data['geo_lon'])
                    )
    return house

def create_street_node(data):
    street = Node("Street",
                    region=int(data['id_region']),
                    street=int(data['street_id']))
    return street

def create_region_node(data):
    region = Node("Region",
                    region=int(data['id_region']),
                    avg_lat=float(data['avg_geo_lat']),
                    avg_lon=float(data['avg_geo_lon']))
    return region

#属性
def create_property_attr_node(data):
    price_node = Node("Price",
                    id=int(data['id']),
                    price=int(data['price']))
    level_node = Node("Level",
                    id=int(data['id']),
                    level=int(data['level']))
    levels_node = Node("Levels",
                    id=int(data['id']),
                    levels=int(data['levels']))
    rooms_node = Node("Rooms",
                    id=int(data['id']),
                    rooms=int(data['rooms']))
    area_node = Node("Area",
                    id=int(data['id']),
                    area=float(data['area']))
    kitchen_area_node = Node("KitchenArea",
                    id=int(data['id']),
                    kitchen_area=float(data['kitchen_area'])
                             )
    return price_node,level_node,levels_node,rooms_node,area_node,kitchen_area_node

def create_house_attr_node(data):
    lat_node = Node("Lat",
                    house=int(data['house_id']),
                    lat=float(data['geo_lat'])
                             )
    lon_node = Node("Lon",
                    house=int(data['house_id']),
                    lon=float(data['geo_lon'])
                             )
    return lat_node,lon_node

def create_street_attr_node(data):
    postcode_node = Node("Postcode",
                    street=int(data['street_id']),
                    postcode=int(data['postal_code'])
                             )
    return postcode_node


def import_house_data(df_property,df_house,df_street,df_region):
    for index, row in tqdm(df_property.iterrows(),desc="building property node") :
        property_node=create_property_node(row)
        price_node, level_node, levels_node, rooms_node, area_node, kitchen_area_node=create_property_attr_node(row)
        graph.create(property_node)
        graph.create(price_node)
        graph.create(level_node)
        graph.create(levels_node)
        graph.create(rooms_node)
        graph.create(area_node)
        graph.create(kitchen_area_node)
    print("property node finished")

    for index, row in tqdm(df_house.iterrows(),desc="building house node") :
        house_node=create_house_node(row)
        lat_node,lon_node=create_house_attr_node(row)
        graph.create(house_node)
        graph.create(lat_node)
        graph.create(lon_node)
    print("house node finished")

    for index, row in tqdm(df_street.iterrows(),desc="building street node") :
        street_node=create_street_node(row)
        postcode_node=create_street_attr_node(row)
        graph.create(street_node)
        graph.create(postcode_node)
    print("street type finished")

    for index, row in tqdm(df_region.iterrows(),desc="building region node") :
        region_node=create_region_node(row)
        graph.create(region_node)
    print("region type finished")

import_house_data(df_property,df_house,df_street,df_region)
# #
# # get relation
matcher = NodeMatcher(graph)
#
for i in tqdm(df_property.values, desc="building IN_HOUSE relation"):
    house_id = int(i[-1])
    id=int(i[1])
    house_node = graph.nodes.match("House", house=house_id).first()
    property_node = graph.nodes.match("Property", id=id).first()
    r = Relationship(property_node, 'LOCATE_IN', house_node)
    graph.create(r)
    r = Relationship(house_node, 'LOCATE', property_node)
    graph.create(r)

    price_nodes = graph.nodes.match("Price", id=id).first()
    r = Relationship(property_node, 'REAL_ESTATE_PRICE', price_nodes)
    graph.create(r)

    levels_nodes = graph.nodes.match("Levels", id=id).first()
    r = Relationship(property_node, 'TOTAL_LAYER', levels_nodes)
    graph.create(r)

    level_nodes = graph.nodes.match("Level", id=id).first()
    r = Relationship(property_node, 'IN_LAYER', level_nodes)
    graph.create(r)

    rooms_nodes = graph.nodes.match("Rooms", id=id).first()
    r = Relationship(property_node, 'HAS_ROOM', rooms_nodes)
    graph.create(r)

    area_nodes = graph.nodes.match("Area", id=id).first()
    r = Relationship(property_node, 'HAS_AREA', area_nodes)
    graph.create(r)

    kitchenarea_nodes = graph.nodes.match("KitchenArea", id=id).first()
    r = Relationship(property_node, 'HAS_KITCHENAREA', kitchenarea_nodes)
    graph.create(r)

for i in tqdm(df_house.values, desc="building IN_STREET relation"):
    street_id = int(i[-1])
    house_id=int(i[1])
    house_node = graph.nodes.match("House", house=house_id).first()

    street_node = graph.nodes.match("Street", street=street_id).first()
    r = Relationship(house_node, 'IN_STREET', street_node)
    graph.create(r)
    r = Relationship(street_node, 'STREET', house_node)
    graph.create(r)

    lat_nodes = graph.nodes.match("Lat", house=house_id).first()
    r = Relationship(house_node, 'POS_LAT', lat_nodes)
    graph.create(r)

    lon_nodes = graph.nodes.match("Lon", house=house_id).first()
    r = Relationship(house_node, 'POS_LON', lon_nodes)
    graph.create(r)

for i in tqdm(df_street.values,desc="building region relation"):
    street_id = int(i[1])
    region_id = int(i[-1])
    street_node = graph.nodes.match("Street", street=street_id).first()

    region_node = graph.nodes.match("Region", region=region_id).first()
    r = Relationship(street_node, 'IN_REGION', region_node)
    graph.create(r)
    r = Relationship(region_node, 'REGION', street_node)
    graph.create(r)

    postalcode_node = graph.nodes.match("Postcode", street=street_id).first()
    r = Relationship(street_node, 'HAS_POSTCODE', postalcode_node)
    graph.create(r)

regions = graph.nodes.match("Region")
for loc1 in tqdm(regions,desc="position NEAR"):
    for loc2 in regions:
        if loc1 != loc2:  # 避免自身与自身比较
            # 获取节点的经纬度坐标
            coords1 = (loc1['avg_lat'], loc1['avg_lon'])
            coords2 = (loc2['avg_lat'], loc2['avg_lon'])
            # 计算地理位置之间的距离（单位：米）
            distance = geodesic(coords1, coords2).meters
            # 如果距离小于阈值，创建关系
            if distance < 1700000:    #市中心的平均值
                relation = Relationship(loc1, "NEAR", loc2, distance=distance)
                graph.create(relation)