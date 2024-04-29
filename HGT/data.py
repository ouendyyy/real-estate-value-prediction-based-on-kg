import pandas as pd
import dgl
import torch
from geopy.distance import geodesic  #给定经纬度，计算两个地理位置的距离
from torch.nn.functional import normalize
def preprocess(property_path,house_path,street_path,region_path):
    df_house = pd.read_csv(house_path,index_col=False)
    df_property = pd.read_csv(property_path,index_col=False)
    df_street = pd.read_csv(street_path,index_col=False)
    df_region= pd.read_csv(region_path,index_col=False)

    df_house = df_house.drop(df_house.columns[0], axis=1)
    df_property = df_property.drop(df_property.columns[0], axis=1)
    df_street = df_street.drop(df_street.columns[0], axis=1)
    df_region = df_region.drop(df_region.columns[0], axis=1)

    df_house_feat = df_house.drop(columns=['house_id','street_id'])
    df_property_feat = df_property.drop(columns=['house_id', 'id','price'])
    df_street_feat = df_street.drop(columns=['id_region', 'street_id'])
    df_region_feat = df_region.drop(columns=['id_region'])
    tensor_house = torch.tensor(df_house_feat.values, dtype=torch.float32)
    tensor_property = torch.tensor(df_property_feat.values, dtype=torch.float32)
    tensor_street = torch.tensor(df_street_feat.values, dtype=torch.float32)
    tensor_region = torch.tensor(df_region_feat.values, dtype=torch.float32)

    class ProjectionLayer(torch.nn.Module):
        def __init__(self, in_sizes, out_size):
            super(ProjectionLayer, self).__init__()
            self.layers = torch.nn.ModuleDict({
                'property' : torch.nn.Linear(in_sizes['property'], out_size),
                'house' : torch.nn.Linear(in_sizes['house'], out_size),
                'street':torch.nn.Linear(in_sizes['street'], out_size),
                'region':torch.nn.Linear(in_sizes['region'], out_size)})
        def forward(self, feats):
            # user and item features can have different lengths but
            # will become the same after the projection layer
            return {'property' : self.layers['property'](feats['property']),
                    'house' : self.layers['house'](feats['house']),
                    'street' : self.layers['street'](feats['street']),
                    'region' : self.layers['region'](feats['region'])}

    in_sizes = {'property': 5, 'house': 2, 'street': 1, 'region': 2}
    out_size = 70

    projection_layer = ProjectionLayer(in_sizes, out_size)

    feats={'property':tensor_property,'house':tensor_house,'street':tensor_street,'region':tensor_region}
    # 使用 ProjectionLayer 进行特征变换
    out_feats = projection_layer(feats)     #特征字典

    df_house['houseid']=range(len(df_house))
    df_street['streetid']=range(len(df_street))
    df_region['regionid'] = range(len(df_region))

    locate_in = pd.merge(df_house, df_property, on='house_id', how='inner')
    in_street = pd.merge(df_house, df_street, on='street_id', how='inner')
    in_region = pd.merge(df_street, df_region, on='id_region', how='inner')
    house_id1 = locate_in['houseid'].tolist()
    property_id = locate_in['id'].tolist()
    street_id1 = in_street['streetid'].tolist()
    house_id2 = in_street['houseid'].tolist()
    street_id2 = in_region['streetid'].tolist()
    region_id = in_region['regionid'].tolist()
    id_regions1 = []
    id_regions2 = []
    for index1,loc1 in df_region.iterrows():
        for index2,loc2 in df_region.iterrows():
            if loc1['regionid'] != loc2['regionid']:  # 避免自身与自身比较
                # 获取节点的经纬度坐标
                coords1 = (loc1['avg_geo_lat'], loc1['avg_geo_lon'])
                coords2 = (loc2['avg_geo_lat'], loc2['avg_geo_lon'])
                # 计算地理位置之间的距离（单位：米）
                distance = geodesic(coords1, coords2).meters
                # 如果距离小于阈值，创建关系
                if distance < 1500000:  # 市中心的平均值
                    id_region1 = loc1['regionid']
                    id_region2 = loc2['regionid']
                    id_regions1.append(id_region1)
                    id_regions2.append(id_region2)

    G = dgl.heterograph({
        ('property', 'LOCATE_IN', 'house'): (property_id,house_id1),
        ('house', 'LOCATE', 'property'): (house_id1, property_id),
        ('house', 'IN_STREET', 'street'): (house_id2,street_id1),
        ('street', 'STREET', 'house'): (street_id1, house_id2),
        ('street', 'IN_REGION', 'region'): (street_id2,region_id),
        ('region', 'REGION', 'street'): (region_id, street_id2),
        ('region', 'NEAR', 'region'): (id_regions1,id_regions2),
        ('region', 'NEAR', 'region'): (id_regions2, id_regions1)
    })
    labels = torch.tensor(df_property['price'], dtype=torch.float)/1000000
    return G,out_feats,labels
