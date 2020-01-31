#IMPORTS--------------------------------------------------------------------------------------------------------------------------------------------------------------------
import os, sys, time, colorsys, heapq, datetime, psutil, sqlite3, json
import itertools as it
import numpy as np
from collections import Counter
from operator import itemgetter
from copy import deepcopy as copy
from scipy import __version__
from scipy.sparse import csr_matrix as csr
from scipy.sparse.csgraph import connected_components
from scipy.sparse import diags, hstack, vstack, triu, isspmatrix_csr
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#GLOBALS--------------------------------------------------------------------------------------------------------------------------------------------------------------------
OBS = 0; CAR = 1; SPE = 2; GEN = 3; REP = 4; STR = 5; MOD = 6; RID = 7; SP_ = 8; TYP = 9; CLU = 0; MER = 1; DIS = 2; SIM = 1;

MEM = [0,0,0]; TIME = [0,0,0]; SIZE = 0; TIME_ = [0,0]; MERGE = 0; CLUS = 0; COMP = np.array([1]); BOUND = 0; MAXF1 = 0.; MAXPR = [0.,0.];

_ftypes_    = {'affiliations':.2,'categories':.18,'coauthornames':.2,'emails':.1,'keywords':.1,'refauthornames':.12,'terms':.15,'years':.02};
_old_scipy_ = int(__version__.split('.')[0])==0;
_p_         = psutil.Process(os.getpid()); _mb_ = 1024*1024;

_checker_   = False; # <---------------------------------------TODO:WARNING !!! This might not be wanted !!!

_surname     =              sys.argv[1];
_firstinit   =              sys.argv[2] if sys.argv[2] != 'None' else None;
_result_db   =              sys.argv[3];
_cfg_file    =              sys.argv[4];
_p_new_      = bool(int(    sys.argv[5]));
_d_          = float(       sys.argv[6]);
_random_     = bool(int(    sys.argv[7]));
_nbrdm_      = True if      sys.argv[8]=='1' else False;
_top_k_      = None if      sys.argv[9]=='0' else int(sys.argv[7]);
_dbscan_     = bool(int(    sys.argv[10]));
_similarity_ = 'probsim' if sys.argv[11]=='0' else 'cosim';

cfg_in = open(_cfg_file,'r'); _cfg = json.loads(cfg_in.read()); cfg_in.close();

_feat_db = _cfg['root_dir']+_cfg['feat_dir']+_surname+'_'+_firstinit+'.db';
_sums_db = _feat_db if _cfg['sums_db'] == None else _cfg['sums_db'];
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-CLASSES-------------------------------------------------------------------------------------------------------------------------------------------------------------------
class DATA:

    def __init__(self,nodes,mentions,rIDs,lat_els,lat_edg,make_match,aggregate):
        #--------------------------------------------------------------------------------------------
        print 'Initializing...'; t = time.time(); # Creating central mappings and rows+cols
        #--------------------------------------------------------------------------------------------
        index2node           = nodes.keys();
        node2index           = {index2node[i]:i for i in xrange(len(index2node))};
        index2rID            = rIDs;
        rID2index            = {index2rID[i]:i for i in xrange(len(index2rID))};
        index2type           = list(lat_els);
        type2index           = {index2type[i]:i for i in xrange(len(index2type))};
        obs, N               = [], [];
        node2type            = [None for node in nodes];
        type2nodes           = [[] for typ in lat_els];
        rows_edge, cols_edge = [], [];
        for gen_str in nodes:
            obs.append(nodes[gen_str][OBS]);
            N.append(nodes[gen_str][CAR]);
            type_index            = type2index[nodes[gen_str][TYP]];
            node_index            = node2index[gen_str];
            node2type[node_index] = type_index;
            type2nodes[type_index].append(node_index);
            for spe_str in nodes[gen_str][SPE]|set([gen_str]):
                rows_edge.append(node_index);
                cols_edge.append(node2index[spe_str]);
        #--------------------------------------------------------------------------------------------
        print time.time()-t, 'seconds for 1.', _p_.memory_info().rss/_mb_, 'MB used'; t = time.time(); # Matching information if required
        #--------------------------------------------------------------------------------------------
        rows_match, cols_match = [], [];
        if make_match:
            for i in xrange(len(nodes.keys())):
                for j in xrange(len(nodes.keys())):
                    str1 = nodes.keys()[i]; str2 = nodes.keys()[j];
                    if matches(nodes[str1][REP],nodes[str2][REP]):
                        rows_match.append(node2index[str1]);
                        cols_match.append(node2index[str2]);
        #--------------------------------------------------------------------------------------------
        print time.time()-t, 'seconds for 2.', _p_.memory_info().rss/_mb_, 'MB used'; t = time.time(); # Creating more mappings
        #--------------------------------------------------------------------------------------------
        mention2node    = [];
        node2mentions   = [[]   for node    in nodes   ];
        rID2nodes       = [[]   for rID     in rIDs    ];
        node2rIDs       = [[]   for node    in nodes   ];
        rID2mentions    = [[]   for rID     in rIDs    ];
        mention2rID     = [None for mention in mentions];
        index2mentionID = [];
        mentionID2index = dict();
        for mention_index in xrange(len(mentions)):
            node_index = node2index[string(mentions[mention_index][0])];
            rID        = mentions[mention_index][1];
            mentionID  = mentions[mention_index][3];
            rID_index  = rID2index[rID];
            mention2node.append(node_index);
            node2mentions[node_index].append(mention_index);
            rID2nodes[rID_index].append(node_index);
            rID2mentions[rID_index].append(mention_index);
            node2rIDs[node_index].append(rID_index);
            mention2rID[mention_index] = rID_index;
            mentionID2index[mentionID] = len(index2mentionID);
            index2mentionID.append(mentionID);
        #--------------------------------------------------------------------------------------------
        print time.time()-t, 'seconds for 3.', _p_.memory_info().rss/_mb_, 'MB used'; t = time.time(); # More rows+cols
        #--------------------------------------------------------------------------------------------
        rows_NM, cols_NM     = zip(*[[node_index,mention_index] for node_index in xrange(len(node2mentions)) for mention_index in node2mentions[node_index]]);
        rows_MR, cols_MR     = zip(*[[mention_index,mention2rID[mention_index]] for mention_index in xrange(len(mention2rID))]);
        rows_TT, cols_TT     = zip(*[[type2index[edg[0]],type2index[edg[1]]] for edg in lat_edg]);
        rows_NT, cols_NT     = zip(*[[node_index,node2type[node_index]] for node_index in xrange(len(node2type))]);
        rows_spec, cols_spec = zip(*[[node2index[gen_str],node2index[spe_str]] for gen_str in nodes for spe_str in nodes[gen_str][SP_]|set([gen_str])]);
        #--------------------------------------------------------------------------------------------
        print time.time()-t, 'seconds for 4.', _p_.memory_info().rss/_mb_, 'MB used'; t = time.time(); # Replace list of mentions by length
        #--------------------------------------------------------------------------------------------
        if not aggregate:
            for key, val in nodes.items():
                nodes[key][RID] = Counter({rid:len(val[RID][rid]) for rid in val[RID]});
        #--------------------------------------------------------------------------------------------
        print time.time()-t, 'seconds for 5.', _p_.memory_info().rss/_mb_, 'MB used'; t = time.time();
        #--------------------------------------------------------------------------------------------
        self.nodes           = nodes;
        self.index2node      = index2node;
        self.node2index      = node2index;
        self.index2rID       = index2rID;
        self.rID2index       = rID2index;
        self.index2mentionID = index2mentionID;
        self.mentionID2index = mentionID2index;
        self.edge            = csr((np.ones(len(rows_edge)),(rows_edge,cols_edge)),shape=(len(self.nodes),len(self.nodes)), dtype=bool);
        self.obs             = csr(np.array(obs).reshape(len(obs),1),shape=(len(obs),1),dtype=float); #TODO: Should this be defined twice?
        self.car             = csr(np.array(N).reshape(len(N),1),shape=(len(N),1),dtype=float);
        self.match           = csr((np.ones(len(rows_match)),(rows_match,cols_match)),shape=(len(self.nodes),len(self.nodes)), dtype=bool);
        self.NM              = csr((np.ones(len(rows_NM)),(rows_NM,cols_NM)),shape=(len(self.nodes),len(mentions)), dtype=bool);
        self.MR              = csr((np.ones(len(rows_MR)),(rows_MR,cols_MR)),shape=(len(mentions),len(rIDs)), dtype=bool);
        self.TT              = csr((np.ones(len(rows_TT)),(rows_TT,cols_TT)),shape=(len(lat_els),len(lat_els)), dtype=bool);
        self.NT              = csr((np.ones(len(rows_NT)),(rows_NT,cols_NT)),shape=(len(self.nodes),len(lat_els)), dtype=bool);
        self.ment            = csr([[mention[2]] for mention in mentions]);
        self.spec            = csr((np.ones(len(rows_spec)),(rows_spec,cols_spec)),shape=(len(self.nodes),len(self.nodes)), dtype=bool);
        #--------------------------------------------------------------------------------------------
        print time.time()-t, 'seconds for 6.', _p_.memory_info().rss/_mb_, 'MB used'; t = time.time();
        #--------------------------------------------------------------------------------------------
        self.obs             = self.NM.dot(self.ment);
        self.MR_             = csr(self.ment).multiply(self.MR);
        self.core            = np.zeros(self.ment.shape[0],dtype=bool);
        self.arrow           = diags(np.ones(self.ment.shape[0],dtype=int),0,dtype=bool);
        self.labels          = np.arange(self.car.shape[0]); # initially each node is one cluster
        self.labelling       = self.NM.T.nonzero()[1];
        self.n               = len(self.labels);
        self.MC              = csr((np.ones(len(self.labelling),dtype=bool),(np.arange(len(self.labelling)),self.labelling)),shape=(len(self.labelling),len(self.labels)),dtype=bool);
        self.NC              = self.NM.dot(self.MC);
        self.rids_c          = self.MC.T.dot(self.MR_);
        NR                   = self.NM.dot(self.MR);
        self.rids_b          = self.NM.dot(self.MR_);
        self.new             = np.ones(self.car.shape[0],dtype=bool);
        self.update_weights();
        #--------------------------------------------------------------------------------------------
        print time.time()-t, 'seconds for 7.', _p_.memory_info().rss/_mb_, 'MB used'; t = time.time();
        #--------------------------------------------------------------------------------------------
        self.index2feat = {ftype: get_index2feat(ftype,mentionID2index,_feat_db) for ftype in _ftypes_};
        self.feat2index = {ftype: {self.index2feat[ftype][i]:i for i in xrange(len(self.index2feat[ftype]))} for ftype in _ftypes_};
        #--------------------------------------------------------------------------------------------
        print time.time()-t, 'seconds for 8.', _p_.memory_info().rss/_mb_, 'MB used'; t = time.time();
        #--------------------------------------------------------------------------------------------
        self.MF         = {ftype: get_MF(ftype,self.mentionID2index,self.feat2index[ftype],_feat_db) for ftype in _ftypes_};
        #--------------------------------------------------------------------------------------------
        print time.time()-t, 'seconds for 9.', _p_.memory_info().rss/_mb_, 'MB used'; t = time.time();
        #--------------------------------------------------------------------------------------------
        self.f          = {ftype: get_f(ftype,self.index2feat[ftype],_sums_db) for ftype in _ftypes_};
        #self.f          = {ftype: np.ravel(self.MF[ftype].sum(0))                 for ftype in _ftypes_}; #TODO: Normalization check
        #--------------------------------------------------------------------------------------------
        print time.time()-t, 'seconds for 10.', _p_.memory_info().rss/_mb_, 'MB used'; t = time.time();
        #--------------------------------------------------------------------------------------------
        self.one_by_f   = {ftype: 1./self.f[ftype] for ftype in _ftypes_};
        #--------------------------------------------------------------------------------------------
        print time.time()-t, 'seconds for 11.', _p_.memory_info().rss/_mb_, 'MB used'; t = time.time();
        #--------------------------------------------------------------------------------------------
        self.freq_x     = {ftype: np.array(self.MF[ftype].sum(1).T) for ftype in _ftypes_};
        #--------------------------------------------------------------------------------------------
        print time.time()-t, 'seconds for 12.', _p_.memory_info().rss/_mb_, 'MB used'; t = time.time();
        #--------------------------------------------------------------------------------------------

    def update_index(self,keep,r,changed):
        self.index2node         = list(itemgetter(*keep)(self.index2node));
        self.node2index         = {self.index2node[i]:i for i in xrange(len(self.index2node))};
        self.new                = self.new[keep];
        self.new[keep.index(r)] = changed;

    def update_weights(self):
        numerator   = self.car if not _cfg['milojevic'] else self.obs;
        denominator = self.car;
        self.weight = numerator.T.multiply(self.edge).multiply(csr(1./denominator.toarray(),shape=denominator.shape,dtype=float));           mem=_p_.memory_info().rss/_mb_; MEM[MER]=max(MEM[MER],mem);
        if _old_scipy_:
            self.weight = set_diagonal(self.weight,self.obs/denominator);#D.weight.setdiag(D.obs.toarray()/D.car.toarray());
        else:
            self.weight = set_diagonal(self.weight,csr(self.obs/denominator,shape=self.obs.shape));#D.weight.setdiag(np.ravel(D.obs/D.car));                                                                                       mem=_p_.memory_info().rss/_mb_; MEM[MER]=max(MEM[MER],mem);
        self.weight.eliminate_zeros();                                                                                                       mem=_p_.memory_info().rss/_mb_; MEM[MER]=max(MEM[MER],mem);

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#FUNCTIONS------------------------------------------------------------------------------------------------------------------------------------------------------------------

#-FUNCTIONS-DRAWING--------------------------------------------------------------------------------------------------------------------

def get_colors(n):
    hsv = [(x/float(n+1), 0.7, 0.999999) for x in range(n+2)];
    return hsv;

def color_string(R_,colors): #R_ is sparse boolean row vector with nonzero count for each rID present
    denom  = R_[0,_cfg['no_none']:].sum(); #index 0 is for None
    string = ':'.join([' '.join([str(num) for num in colors[i]])+';'+str(round(R_[0,i]/denom,4)) for i in R_[0,_cfg['no_none']:].nonzero()[1]]); #index 0 is for None
    return string;

def get_nodes_edges(D,colors):
    edges    = [];
    nodestrs = [];
    str2dis = dict();
    for i in xrange(len(D.index2node)):
        mentionIDIndeces = D.NM[i,:].nonzero()[1];
        clusterIndeces   = D.NC[i,:].nonzero()[1];
        bPrec,bRec,bF1   = prec_rec_f1(D.rids_b[i,1:]);
        dPrec,dRec,dF1   = prec_rec_f1(D.rids_c[clusterIndeces,1:]) if len(clusterIndeces)>0 else [1.0,1.0,1.0];
        bPrec_gen        = int(round(bPrec,2)*100) if D.nodes[D.index2node[i]][RID].keys() != [] and D.nodes[D.index2node[i]][RID].keys() != [None] else '';
        bRec_gen         = int(round(bRec ,2)*100) if D.nodes[D.index2node[i]][RID].keys() != [] and D.nodes[D.index2node[i]][RID].keys() != [None] else '';
        dPrec_gen        = int(round(dPrec,2)*100) if D.nodes[D.index2node[i]][RID].keys() != [] and D.nodes[D.index2node[i]][RID].keys() != [None] else '';
        dRec_gen         = int(round(dRec ,2)*100) if D.nodes[D.index2node[i]][RID].keys() != [] and D.nodes[D.index2node[i]][RID].keys() != [None] else '';
        color_str        = color_string(D.rids_b[i],colors);
        #node_display     = '"'+str(round(D.obs[i,0],1))+'  |  '+str(round(D.car[i,0],1))+'\n'+D.index2node[i]+'\n ('+str(len(mentionIDIndeces))+')'+str(bPrec_gen)+'/'+str(dPrec_gen)+'|'+str(dRec_gen)+' ('+str(len(clusterIndeces))+')"';
        node_display     = '"'+str(round(D.obs[i,0],1))+'  |  '+str(round(D.car[i,0],1))+'\n'+D.index2node[i]+'"';
        str2dis[D.index2node[i]] = node_display;
        if D.rids_b[i].sum() != 0: nodestrs.append(node_display+' [style=striped fillcolor="'+color_str+'"]');
        for j in D.edge[i,:].nonzero()[1]:
            mentionIDIndeces = D.NM[j,:].nonzero()[1];
            clusterIndeces   = D.NC[j,:].nonzero()[1];
            bPrec,bRec,bF1   = prec_rec_f1(D.rids_b[j,1:]);
            dPrec,dRec,dF1   = prec_rec_f1(D.rids_c[clusterIndeces,1:]) if len(clusterIndeces)>0 else [1.0,1.0,1.0];
            bPrec_spe        = int(round(bPrec,2)*100) if D.nodes[D.index2node[j]][RID].keys() != [] and D.nodes[D.index2node[j]][RID].keys() != [None] else '';
            bRec_spe         = int(round(bRec ,2)*100) if D.nodes[D.index2node[j]][RID].keys() != [] and D.nodes[D.index2node[j]][RID].keys() != [None] else '';
            dPrec_spe        = int(round(dPrec,2)*100) if D.nodes[D.index2node[j]][RID].keys() != [] and D.nodes[D.index2node[j]][RID].keys() != [None] else '';
            dRec_spe         = int(round(dRec ,2)*100) if D.nodes[D.index2node[j]][RID].keys() != [] and D.nodes[D.index2node[j]][RID].keys() != [None] else '';
            #child_display    = '"'+str(round(D.obs[j,0],1))+'  |  '+str(round(D.car[j,0],1))+'\n'+D.index2node[j]+'\n ('+str(len(mentionIDIndeces))+')'+str(bPrec_spe)+'/'+str(dPrec_spe)+'|'+str(dRec_spe)+' ('+str(len(clusterIndeces))+')"';
            child_display    = '"'+str(round(D.obs[j,0],1))+'  |  '+str(round(D.car[j,0],1))+'\n'+D.index2node[j]+'"';
            edges.append(child_display+' -> '+node_display+'[label="'+str(round(D.weight[i,j],2)).strip('0')+'" dir="back"]');
    return nodestrs, edges, str2dis;

def draw(D,colors,I=0):
    OUT = open(_cfg['viz_file']+'.'+str(I),'w') if I != 0 else open(_cfg['viz_file'],'w');
    OUT.write('digraph G {\nranksep=.5\nnode [shape=box]\n');
    nodestrs, edges, str2dis = get_nodes_edges(D,colors);
    for edge in edges:
        OUT.write(edge+'\n');
    for nodestr in nodestrs:
        OUT.write(nodestr+'\n');
    #prec, rec, f1 = prec_rec_f1_([D.nodes[i][RID] for i in D.node2index]);
    bPrec, bRec, bF1 = prec_rec_f1(D.rids_b[:,1:]);
    dPrec, dRec, dF1 = prec_rec_f1(D.rids_c[:,1:]);
    print 'bPrec:',bPrec,'bRec:',bRec,'bF1:',bF1;
    print 'dPrec:',dPrec,'dRec:',dRec,'dF1:',dF1;
    #OUT.write('"bPrec:  '+str(round(bPrec,2))+'\n\nbRec:  '+str(round(bRec,2))+'\n\nbF1:  '+str(round(bF1,2))+'\n\ndPrec:  '+str(round(dPrec,2))+'\n\ndRec:  '+str(round(dRec,2))+'\n\ndF1:  '+str(round(dF1,2)) +'" [style=filled fillcolor=grey fontsize=18]\n');
    #nodes_by_level = get_nodes_by_level(nodes);
    #for level in nodes_by_level:
    #    OUT.write('{rank=same; ');
    #    for node_str in nodes_by_level[level]:
    #        OUT.write(str2dis[node_str]+'; ');
    #    OUT.write('}');
    OUT.write('}');
    OUT.close();

def output(D,I,B,thr_iter,t_start,m_time,c_time,con_out,cur_out):
    global MEM, TIME_, SIZE;
    B              = B.split('.')[0]+', p_new:'+str(_p_new_)+', disc.:'+str(_d_)+', random:'+str(_cfg['num_rdm'])+'/'+str(_nbrdm_)+', step:'+str(_cfg['step']);
    t_iter         = datetime.datetime.utcnow().isoformat().replace('T',' ')[:-7];
    tp_b, p_b, t_b = tp_p_t(D.rids_b[:,1:]);
    tp_c, p_c, t_c = tp_p_t(D.rids_c[:,1:]);
    P_b, R_b, F1_b = [round(val*100,0) for val in prec_rec_f1(D.rids_b[:,1:])];
    P_c, R_c, F1_c = [round(val*100,0) for val in prec_rec_f1(D.rids_c[:,1:])];
    num_m, num_c   = D.MC.shape;
    num_b, num_r   = D.rids_b.shape;
    blocksum       = (D.NM.dot(D.MR[:,1:].astype(int)).sum(1).A**2).sum();#(D.NM.sum(1).A**2).sum();
    general_params = (t_start,t_iter,B,_cfg['eps'],_cfg['z'],_cfg['r'],_cfg['min_pts'],thr_iter,I);#_cfg['smooth']
    statistics     = (num_m,num_r,MERGE,CLUS,(float(COMP.sum())/(COMP>=1).sum()),blocksum,BOUND);
    performance_b  = (num_b,P_b,R_b,F1_b,tp_b,t_b,p_b);
    performance_c  = (num_c,P_c,R_c,F1_c,tp_c,t_c,p_c,MAXPR[0],MAXPR[1],MAXF1);
    cost           = (SIZE,MEM[CLU],MEM[MER],MEM[DIS],round(TIME[CLU],2),round(TIME_[SIM],2),round(TIME_[CLU],2),round(TIME[MER],2),round(TIME[DIS],2),round(m_time,2),round(c_time,2));
    values         = general_params + statistics + performance_b + performance_c + cost; #print values;
    cur_out.execute("INSERT INTO results VALUES("+','.join(['?' for i in xrange(len(values))])+")",values);
    MEM=[0,0,0]; TIME_=[0,0]; SIZE=0; #print COMP;
    con_out.commit();
#-------------------------------------------------------------------------------------------------------------------------------------
#-FUNCTIONS-UTILS---------------------------------------------------------------------------------------------------------------------

def analyse_sim(D,num=1000):
    pairs  = D.MR[:,1:].dot(D.MR[:,1:].T);
    pairs_ = zip(*pairs.nonzero());
    l      = np.array([probsim(np.array([x]),D,np.array([y]))[0,0] for x,y in pairs_[:min(num,len(pairs_))]]);
    print l;
    print l.max(), l.min(), l.sum()/l.shape[0];

def get_index2feat(ftype,mentionID2index,db):
    print ftype;
    con   = sqlite3.connect(db); cur = con.cursor();
    feats = list(set([feat[0] for mentionID in mentionID2index for feat in cur.execute("SELECT feat FROM "+ftype+" WHERE mentionIDIndex=?",(mentionID,))]));
    con.close();print len(feats),'features'
    return feats;

def get_MF(ftype,mentionID2index,feat2index,db):
    con = sqlite3.connect(db); cur = con.cursor();
    ROWS, COLS, DATA = [], [], [];print ftype,len(feat2index)#, max(feat2index.values()), max(feat2index.keys())
    for mentionID in mentionID2index:
        for feat,freq in cur.execute("SELECT feat,freq FROM "+ftype+" WHERE mentionIDIndex=?",(mentionID,)):
            ROWS.append(mentionID2index[mentionID]); COLS.append(feat2index[feat]); DATA.append(freq);
    con.close();
    return csr((DATA,(ROWS,COLS)),shape=(len(mentionID2index),len(feat2index)),dtype=float);

def get_f(ftype,index2feat,db):
    con = sqlite3.connect(db); cur = con.cursor();
    f   = np.array([cur.execute("SELECT freq FROM "+ftype+"_sums WHERE feat=?",(feat,)).fetchall()[0][0] for feat in index2feat],dtype=float);
    con.close();
    return f;

def set_new(matrix,rs,new,COL):
    matrix.eliminate_zeros();
    rows, cols         = matrix.nonzero();
    data               = matrix.data;
    old                = np.logical_not(np.in1d( [rows,cols][COL] ,rs));
    rows_old, cols_old = rows[old], cols[old];
    data_old           = data[old];
    rows_new, cols_new = new.nonzero();
    if COL:
        cols_new = rs[cols_new];
    else:
        rows_new = rs[rows_new];
    data_new           = new[new!=0];#data_new           = np.ravel(new)[ [cols_new,rows_new][COL] ];
    cols_, rows_       = np.concatenate([cols_old,cols_new],0), np.concatenate([rows_old,rows_new],0);
    data_              = np.concatenate([data_old,data_new],0);
    return csr((data_,(rows_,cols_)),shape=matrix.shape);

def set_diagonal(matrix,new): #WARNING: new is expected to be sparse csr matrix (as opposed to what is expected in set_new)
    matrix.eliminate_zeros(); new.eliminate_zeros();
    rows, cols         = matrix.nonzero();
    data               = matrix.data;
    old                = rows!=cols;
    rows_old, cols_old = rows[old], cols[old];
    data_old           = data[old];
    rows_cols_new      = new.nonzero()[0];
    data_new           = new.data;
    cols_, rows_       = np.concatenate([cols_old,rows_cols_new],0), np.concatenate([rows_old,rows_cols_new],0);
    data_              = np.concatenate([data_old,data_new],0);
    return csr((data_,(rows_,cols_)),shape=matrix.shape);

def prec_rec_f1(rids):
    tp, p, t = [float(val) for val in tp_p_t(rids)];
    if p == 0 or t == 0: return 1.0,1.0,1.0;
    return tp/p, tp/t, 2*((tp/p)*(tp/t))/((tp/p)+(tp/t));

def tp_p_t(rids): #Assumes that you pass one block, not a block partition
    tp = rids.multiply(rids).sum();#rids.power(2).sum();
    p  = np.power(rids.sum(1),2).sum();
    t  = np.power(rids.sum(0),2).sum(1)[0,0];
    return tp, p, t;

def string(node_rep):
    if _cfg['is_names']:
        fields = set([tup[0] for tup in node_rep]);
        return ' '.join([tup[1] for tup in sorted(list(node_rep)) if not((tup[0]=='l' and 'l_' in fields) or (tup[0]=='f1' and 'f1_' in fields) or (tup[0]=='f2' and 'f2_' in fields) or (tup[0]=='f3' and 'f3_' in fields))]);
    return '{'+','.join([tup[1] for tup in sorted(list(node_rep))])+'}';

def list2string(list_rep,fields):
    string = '';
    for i in xrange(len(list_rep)):
        if list_rep[i] != None:
            string += fields[i]+'&"'+list_rep[i]+'";';
    return string[:-1];

def load_node_infos_db(dbfile,surname,firstinit):
    fields     = ['l','l_','f1','f1_','f2','f2_','f3','f3_'] if not _cfg['milojevic'] else ['l','l_','f1','f2','f3'];
    node_infos = [];
    temp_dict  = dict();
    con        = sqlite3.connect(dbfile);
    cur        = con.cursor();    
    if firstinit == None:
        cur.execute("SELECT mentionIDIndex, rIDIndex, "+', '.join(fields)+" FROM names WHERE l_=?",(surname,));
    else:
        cur.execute("SELECT mentionIDIndex, rIDIndex, "+', '.join(fields)+" FROM names WHERE l_=? AND f1=?",(surname,firstinit,));
    for row in cur:
        mentionID = row[0]#str(row[0]); #TODO: Remember that now the mentionID is expected to be INTEGER!
        rID       = str(row[1]) if row[1] != None else None;
        list_rep  = row[2:];
        key_rep   = list2string(list_rep,fields);
        set_rep   = set([(fields[i],list_rep[i],) for i in xrange(len(fields)) if list_rep[i] != None]);
        if not key_rep in temp_dict:
            temp_dict[key_rep] = [set_rep,Counter({rID:[mentionID]})];
        else:
            if not rID in temp_dict[key_rep][1]:
                temp_dict[key_rep][1][rID]  = [mentionID];
            else:
                temp_dict[key_rep][1][rID] += [mentionID];
    node_infos = [temp_dict[key_rep] for key_rep in temp_dict];
    con.close();
    return node_infos;

def load_lattice(latfile):
    lat_els = set([]);
    lat_edg = [];
    IN = open(latfile,'r');
    for line in IN:
        line_ = tuple(sorted(line.rstrip().split(',')));
        line_ = tuple([]) if line_ == ('',) else line_;
        lat_els.add(line_);
    IN.close();
    for spe_tup in lat_els:
        spe_tup_ = list(spe_tup);
        for i in xrange(len(spe_tup_)):
            gen_tup = tuple(spe_tup_[:i]+spe_tup_[i:]);
            if gen_tup in lat_els:
                lat_edg.append((gen_tup,spe_tup,));
    return lat_els, lat_edg;

def make_node(node_info,aggregate):
    node = [sum(node_info[1].values()),0.0,set([]),set([]),node_info[0],string(node_info[0]),None,node_info[1],set([]),get_type(node_info[0])] if aggregate else [sum([len(lst) for lst in node_info[1].values()]),0.0,set([]),set([]),node_info[0],string(node_info[0]),None,node_info[1],set([]),get_type(node_info[0])];
    return node;

def get_nodes_by_level(nodes):
    nodes_by_level = dict();
    for spe_str in nodes:
        level = len(nodes[spe_str][REP]);
        if level in nodes_by_level:
            nodes_by_level[level].add(spe_str);
        else:
            nodes_by_level[level] = set([spe_str]);
    return nodes_by_level;

def get_type(node_rep):
    return tuple(sorted([el[0] for el in node_rep]));

def sanity_check(D):
    for i in xrange(len(D.index2node)):
        car_cnt = D.car[i,0];
        sum_cnt = D.obs.toarray()[np.ravel(D.spec[i].toarray())].sum();
        if abs(car_cnt-sum_cnt) > 0.000000001:
            print '###WARNING!', D.index2node[i], 'sum:', sum_cnt, 'vs.', car_cnt;
            print 'specifications:', [D.index2node[j] for j in D.spec[i].nonzero()[1]];
            print '--------------------------------------------------------';
#-------------------------------------------------------------------------------------------------------------------------------------
#-FUNCTIONS-COMPARISON----------------------------------------------------------------------------------------------------------------

def licenced(node_rep):
    typ = get_type(node_rep);
    if typ in lat_els:
        return True;
    return False;

def matches(rep1,rep2):
    return licenced(rep1|rep2);

def generalizes(rep1,rep2): #generalizes itself, too
    return len(rep1-rep2)==0;

def frob_for(ftype,xs,D,xs_=np.array([]),MAX=False):
    global MEM;
    xs_      = xs_ if xs_.any() else xs;
    one_by_f = csr(D.one_by_f[ftype]);
    p_x_f    = D.MF[ftype][xs_,:].multiply(one_by_f*_cfg['hack']).tocsr();     mem=_p_.memory_info().rss/_mb_; MEM[CLU]=max(MEM[CLU],mem); #print mem,'MB used after p_x_f';#TODO:change back!
    N        = D.MF[ftype].shape[0];                                    mem=_p_.memory_info().rss/_mb_; MEM[CLU]=max(MEM[CLU],mem); #print mem,'MB used after N';
    num      = D.MF[ftype][xs,:].dot(p_x_f.T).toarray()+_cfg['smooth']/N;     mem=_p_.memory_info().rss/_mb_; MEM[CLU]=max(MEM[CLU],mem); #print mem,'MB used after num';
    f_x_x    = num if not MAX else np.maximum(num,num.T);
    return f_x_x;

def prob_for(ftype,xs,D,xs_=np.array([]),MAX=False):
    xs_ = xs_ if xs_.any() else xs;
    f_x_x = frob_for(ftype,xs,D,xs_,MAX);
    f_x   = D.freq_x[ftype][:,xs_]+_cfg['smooth'];
    p_x_x = np.array(f_x_x / f_x);
    return p_x_x;

def probsim(xs,D,xs_=np.array([]),ftypes=None,MAX=False):
    global TIME_;
    xs_    = xs_    if xs_.any() else xs;
    ftypes = ftypes if ftypes != None else D.MF.keys();
    #print 'similarity';
    t_sim      = time.time();
    similarity = np.zeros((len(xs),len(xs_)),dtype=float);
    for ftype in ftypes:
        #print ftype;
        p_x_x = prob_for(ftype,xs,D,xs_,MAX);
        similarity += p_x_x*(1./len(ftypes));
        del p_x_x;
    TIME_[SIM] += time.time()-t_sim;
    #print 'end similarity';
    return similarity;

def cosine(ftype,xs,D,xs_=np.array([])): #TODO:Smoothing required?
    xs_    = xs_ if xs_.any() else xs;
    num    = D.MF[ftype][xs,:].dot(D.MF[ftype][xs_,:].T).toarray();
    norm   = np.sqrt(D.MF[ftype][xs,:].multiply(D.MF[ftype][xs,:]).sum(1));
    norm_  = np.sqrt(D.MF[ftype][xs_,:].multiply(D.MF[ftype][xs_,:]).sum(1));
    denom  = norm*norm_.T;
    result = np.nan_to_num(num/denom);
    return result.A;    

def cosim(xs,D,xs_=np.array([]),ftypes=None,MAX=False):
    global TIME_;
    xs_        = xs_    if xs_.any() else xs;
    ftypes     = ftypes if ftypes != None else D.MF.keys();
    t_sim      = time.time();
    similarity = np.zeros((len(xs),len(xs_)),dtype=float);
    for ftype in ftypes:
        result      = cosine(ftype,xs,D,xs_);
        similarity += result*(1./len(ftypes));
        del result;
    TIME_[SIM] += time.time()-t_sim;
    return similarity;

def euclidist(xs,D,xs_=np.array([]),ftypes=None):
    xs_    = xs_    if xs_.any() else xs;
    ftypes = ftypes if ftypes != None else D.MF.keys();
    euclid = dict(); #print 'similarity';
    for ftype in ftypes:
        euclid[ftype] = pdist(D.MF[ftype][xs,:],metric='euclidean');#print ftype;
    similarity = np.sum([euclid[ftype]*1./len(euclid) for ftype in euclid],0); #print 'end similarity';
    return similarity;

def sim(xs,D,xs_=np.array([])):
    if _similarity_ == 'probsim':   return probsim(xs,D,xs_);
    if _similarity_ == 'euclidist': return euclidist(xs,D,xs_);
    if _similarity_ == 'cosim':     return cosim(xs,D,xs_);

def reach(similarity):
    N         = similarity.shape[0];
    threshold = get_threshold(N);
    reachable = similarity <= threshold if _similarity_ in ['euclidist'] else similarity >= threshold;
    np.fill_diagonal(reachable,True);
    return reachable;
#-------------------------------------------------------------------------------------------------------------------------------------
#-FUNCTIONS-BUILDING------------------------------------------------------------------------------------------------------------------

def insert(spe_rep,spe_str,count,seen,nodes):
    for tup in spe_rep:
        gen_rep = spe_rep - set([tup]);
        if not licenced(gen_rep): continue;
        gen_str = string(gen_rep);
        if gen_str in nodes:
            nodes[spe_str][GEN].add(gen_str);
            nodes[gen_str][SPE].add(spe_str);
            nodes[gen_str][SP_] |= seen|set([spe_str]);
            if nodes[gen_str][MOD] != iteration:
                nodes[gen_str][MOD]  = iteration;
                nodes[gen_str][CAR] += count;
                insert(gen_rep,gen_str,count,seen|set([spe_str]),nodes);
        else:
            nodes[gen_str] = [0,count,set([spe_str]),set([]),gen_rep,gen_str,iteration,Counter(),seen|set([spe_str]),get_type(gen_rep)];
            nodes[spe_str][GEN].add(gen_str);
            insert(gen_rep,gen_str,count,seen|set([spe_str]),nodes);

def add_node(spe_rep,rids,aggregate,nodes):
    global iteration;
    iteration += 1;
    spe_str    = string(spe_rep);
    count      = sum(rids.values()) if aggregate else sum([len(lst) for lst in rids.values()]);
    if spe_str in nodes:
        nodes[spe_str][OBS] += count;
        nodes[spe_str][CAR] += count if not _cfg['milojevic'] else 0;
        nodes[spe_str][RID] += rids;
        nodes[spe_str][MOD]  = iteration;
    else:
        nodes[spe_str] = [count,count,set([]),set([]),spe_rep,spe_str,iteration,rids,set([]),get_type(spe_rep)] if not _cfg['milojevic'] else [count,0,set([]),set([]),spe_rep,spe_str,iteration,rids,set([]),get_type(spe_rep)];
    insert(spe_rep,spe_str,count,set([]),nodes);
#-------------------------------------------------------------------------------------------------------------------------------------
#-FUNCTIONS-MODIFICATION--------------------------------------------------------------------------------------------------------------

def combine(matrix,group,r,keep,reach=False):
    t_ = time.time(); #print 'Start shape:', matrix.shape;
    # Representant gets combination of values of group | Sets the rth row to be the sum of the group rows
    matrix = set_new(matrix,np.array([r]),matrix[group,:].toarray().sum(0,matrix.dtype)[None,:],False);
    #print 'A', time.time()-t_, matrix.shape, len(matrix.nonzero()[0]), reach; t = time.time();
    # If the matrix is quadratic (D.edge, D.spec), then whatever goes to group, also goes to r | Sets the rth column to be the sum of the group columns
    if matrix.shape[0] == matrix.shape[1]:
        matrix = set_new(matrix,np.array([r]),matrix[:,group].toarray().sum(1,matrix.dtype)[:,None],True);
    #print 'B', time.time()-t, matrix.shape, len(matrix.nonzero()[0]), reach; t = time.time();
    # If this applies (D.spec), whatever reaches r, now also reaches what r reaches | Adds rth row to all rows with 1 in rth column
    if reach:
        reaches_r = D.spec[:,r].nonzero()[0];
        if len(reaches_r) != 0:
            matrix = set_new(matrix,reaches_r,matrix[reaches_r,:].toarray()+matrix[r,:].toarray(),False);
    #print 'C', time.time()-t, matrix.shape, len(matrix.nonzero()[0]), reach; t = time.time();
    # Everything in group except representant gets their values removed | Makes the matrix smaller
    if matrix.shape[0] == matrix.shape[1]:
        matrix = matrix[keep,:][:,keep];
    else:
        matrix = matrix[keep,:];
    #print 'D', time.time()-t, matrix.shape, len(matrix.nonzero()[0]), reach; t = time.time();
    #print 'E', time.time()-t; t = time.time();
    #print 'Combined. Took', time.time()-t_, 'seconds for', matrix.shape;
    return matrix;

def components(edges,core,oldlabels): #TODO: Check if edges should be sparse
    global MEM, TIME_;
    t_clu     = time.time();
    label     = 0; #print 'components';
    edges     = csr(edges,dtype=bool,shape=edges.shape);
    labelling = np.array(xrange(len(core),len(core)*2),dtype=int);
    remaining = np.copy(core);
    reachable = np.zeros(len(core),dtype=bool);
    visited   = np.zeros(len(core),dtype=bool);
    while remaining.any():
        #print 'DBSCAN iteration...';
        if not reachable.any():
            start            = remaining.argmax();#print 'Nothing reachable, start with remaining no.\n', start;
            label            = oldlabels[start];#print 'This one used to have label:\n', label;
            reachable[start] = True;
        else:
            visited += reachable;#print 'So far we have visited:',visited.nonzero()[0]; print np.in1d(visited.nonzero()[0],remaining.nonzero()[0]).nonzero(); print edges[start,start];
        #print 'Reachable before taking closure:', reachable.nonzero()[0]; print 'Remaining node in visited?', np.in1d(visited.nonzero()[0],remaining.nonzero()[0]).nonzero()[0]; print 'Node reaches itself?', edges[start,start];
        #print csr(reachable).shape, edges.shape;
        reachable            = np.ravel(csr(reachable).dot(edges).toarray()) > visited; mem=_p_.memory_info().rss/_mb_; MEM[CLU]=max(MEM[CLU],mem); #print mem,'MB used'; print 'Add all unvisited nodes reachable from what was last reached:\n', reachable.nonzero()[0]; print 'Start remaining?',remaining[start],'reachable?', reachable[start];
        labels               = np.unique(oldlabels[reachable]);#print 'This set of reachable nodes used to have one of the labels\n', labels;
        reachable            = (reachable + np.in1d(oldlabels,labels)) > visited; mem=_p_.memory_info().rss/_mb_; MEM[CLU]=max(MEM[CLU],mem); #print mem,'MB used'; print 'Fast Forward: Add all nodes that used to have one of these labels:\n', reachable.nonzero()[0];
        labelling[reachable] = label;#print 'Make new labelling:\n', labelling; print 'Start remaining?',remaining[start],'reachable?', reachable[start];
        remaining            = remaining > reachable; #print 'Remaining is what was remaining before and has not been reached:\n', remaining.nonzero()[0];
    visited             += reachable;
    reachable            = np.ravel(csr(reachable).dot(edges).toarray()) > visited; mem=_p_.memory_info().rss/_mb_; MEM[CLU]=max(MEM[CLU],mem); #print mem,'MB used';#TODO: Should I remove these visited?
    labelling[reachable] = label;
    labels, labelling    = np.unique(labelling,return_inverse=True); #print 'end components';
    TIME_[CLU] += time.time() - t_clu;
    return len(labels), labelling;

def DBSCAN(D,local):
    # Compute the similarities and reachability for the local context
    sim_loc               = sim(local,D);   #TODO: See if it makes sense to keep global sim table and reuse the old sim, requires mem
    reach_loc             = reach(sim_loc);
    # Compute the core property for the local context
    core_loc              = np.ravel(reach_loc.sum(1) >= _cfg['min_pts']);#print csr(core_loc[None,:]).shape, reach_loc.shape;
    # Compute the arrows for the local context
    arrow_loc             = core_loc[:,None] * reach_loc;
    # Cluster the mentions in the local context
    n_loc, labelling_loc  = components(arrow_loc,core_loc,D.labelling[local]);
    # Integrate the new local labelling into the global context
    labelling_new         = labelling_loc+D.n;
    #labelling             = copy(D.labelling);
    D.labelling[local]    = labelling_new;
    # Update the global labelling and global n
    D.labels, D.labelling = np.unique(D.labelling,return_inverse=True);
    D.n                   = len(D.labels);
    return D;

def logistic(t,G,k,f0):
    return G/( 1 + (np.e**(-k*G*t) * ((G/f0)-1)) );

def root(x,s,n,k):
    return (s*(x**(1.0/n)))-k;

def logist_2(x,h,m,s):
    return logistic(x,h,h,s) + logistic(x,h,h/m,(s/(m*2000.)));

def get_threshold(N):
    if _cfg['tuning']:         return 0.0;
    if _dbscan_:               return _cfg['eps'];
    if _cfg['thr_f']=='root':  return root(    N,_cfg['z'],_cfg['r'],_cfg['eps']);
    if _cfg['thr_f']=='logi':  return logistic(N,_cfg['z'],_cfg['r'],_cfg['eps']);
    if _cfg['thr_f']=='2logi': return logist_2(N,_cfg['z'],_cfg['r'],_cfg['eps']);

def visualize(rids_c): #TODO: Need to find a way to show Recall deficits
    rids_c = rids_c.toarray();
    select = rids_c[:,1:][rids_c[:,1:].sum(1)>0,:];
    string = str([[el for el in line if el != 0] for line in select]);
    return string;

def AGGLO(D,local):   
    global MAXF1, MAXPR;
    MAXF1, MAXPR = 0., [0.,0.];
    #-Compute the iteration-independent components-----------------------------------------------------------------
    N          = len(local);
    C          = N;
    ftypes     = D.MF.keys();
    threshold  = get_threshold(N);
    old_string = '';
    #--------------------------------------------------------------------------------------------------------------
    #-Initialize the iteration-dependent components----------------------------------------------------------------
    MC     = np.identity(N,bool);                                                           # dense
    f_C_C  = np.array(np.concatenate([frob_for(ftype,local,D)[:,:,None] for ftype in ftypes],axis=2)); #(C,C',ftype)
    #f_C    = np.concatenate([np.array((D.MF[ftype][local,:].T.sum(0)+_cfg['smooth'])[:,:,None]) for ftype in ftypes],axis=2); #(1,C',ftype)
    f_C    = np.concatenate([np.array((D.freq_x[ftype][:,local]+_cfg['smooth'])[:,:,None]) for ftype in ftypes],axis=2)
    p_C_C  = (f_C_C / f_C);
    scores = p_C_C.sum(2) / len(ftypes); #print scores.shape;
    np.fill_diagonal(scores,0);
    #--------------------------------------------------------------------------------------------------------------
    #-Tuning-relevant measures-------------------------------------------------------------------------------------
    min_sim = 1.0; max_f1 = 0.0; stamp = datetime.datetime.utcnow().isoformat().replace('T',' ');
    #--------------------------------------------------------------------------------------------------------------
    while C > 1:
        #print p_C_C.sum(0);
        rids_c = csr(MC).T.dot(D.MR_[local,:]);
        #string = visualize(rids_c);
        prec, rec, f1 = prec_rec_f1(rids_c[:,1:]);
        if f1 > MAXF1: MAXF1, MAXPR = [f1, [prec,rec]];
        #-Get the pair with the highest probability----------------------------------------------------------------
        max_pos      = np.unravel_index(np.argmax(scores),scores.shape);
        max_val      = scores[max_pos];
        keep, remove = [[max_pos[0]],[max_pos[1]]] if max_pos[0]<max_pos[1] else [[max_pos[1]],[max_pos[0]]];
        #-Merge the clusters or terminate--------------------------------------------------------------------------
        if max_val < threshold: break;
        if max_val < min_sim: min_sim = max_val;
        if f1 > max_f1:
            max_f1 = f1;
            size_l,size_r = MC[:,max_pos].sum(0);
            cur_out.execute("INSERT INTO tuning VALUES(?,?,?,?,?,?)",(stamp,min_sim,max_f1,N,size_l,size_r,));
        #if string != old_string:
        #    print string; old_string = string;
        #    print '--------------------------------------------------------------------------------------';
        #    print 'P:',int(round(prec,2)*100),'F:',int(round(f1,2)*100),'R:',int(round(rec,2)*100),'|',N,'|',C,'|',MC[:,max_pos[0]].sum(),'+',MC[:,max_pos[1]].sum(),'|',int(rids_c[max_pos[0],1:].sum()),'+',int(rids_c[max_pos[1],1:].sum()),'|', max_val, '>=', threshold;
        #    print '--------------------------------------------------------------------------------------';
        C           -= len(remove);
        MC[:,keep]  += MC[:,remove];
        MC[:,remove] = 0;
        #-Update the iteration-dependent components----------------------------------------------------------------
        f_C[:,keep,:]    += f_C[:,remove,:];
        f_C_C[:,keep,:]  += f_C_C[:,remove,:];
        f_C_C[keep,:,:]  += f_C_C[remove,:,:];
        f_C_C[:,remove,:] = 0;
        f_C_C[remove,:,:] = 0;
        p_C_C[:,remove,:] = 0;
        p_C_C[remove,:,:] = 0;
        p_C_C[:,keep,:]   = (f_C_C[:,keep,:] / f_C[:,keep,:]);
        p_C_C[keep,:,:]   = (f_C_C[keep,:,:] / f_C[:,:,:]);
        scores[:,keep]    = (p_C_C[:,keep,:].sum(2)) / len(ftypes);
        scores[keep,:]    = (p_C_C[keep,:,:].sum(2)) / len(ftypes);
        scores[:,remove]  = 0;
        scores[remove,:]  = 0;
        scores[keep,keep] = 0;
        #print 'scores__________________________________________________________________________'; print scores; print '________________________________________________________________________________';
        #----------------------------------------------------------------------------------------------------------
    rids_c = csr(MC).T.dot(D.MR_[local,:]);
    #string = visualize(rids_c);
    prec, rec, f1 = prec_rec_f1(rids_c[:,1:]);
    if f1 > MAXF1: MAXF1, MAXPR = [f1, [prec,rec]];
    #print string;
    #print '--------------------------------------------------------------------------------------';
    #print 'P:',int(round(prec,2)*100),'F:',int(round(f1,2)*100),'R:',int(round(rec,2)*100),'|',N,'|',C,'|';
    #print '--------------------------------------------------------------------------------------';
    #-Do the remaining standard operations-------------------------------------------------------------------------
    labelling_loc         = MC.nonzero()[1]; # Since we use unique, it shouldn't matter that some cluster-indices are unassigned
    labelling_new         = labelling_loc+D.n; #print labelling_loc; print labelling_new;
    D.labelling[local]    = labelling_new;
    D.labels, D.labelling = np.unique(D.labelling,return_inverse=True);
    D.n                   = len(D.labels); #print 'Made', len(set(labelling_loc)), 'clusters.';
    #--------------------------------------------------------------------------------------------------------------
    #print '--------------------------------------------------------------------------------------\n--------------------------------------------------------------------------------------';
    return D;

def clusterer(D):
    global TIME, SIZE, CLUS, COMP;
    t_clu     = time.time()
    new_nodes = D.new.nonzero()[0];
    for new_node in new_nodes:
        mentions        = D.NM[new_node,:].nonzero()[1];
        SIZE            = max(SIZE,len(mentions));
        COMP[mentions] += 1; #TODO: reset
        print 'Clustering the new node', D.index2node[new_node], 'with', len(mentions), 'mentions';#'with mentions',[D.index2mentionID[mentionIDIndex] for mentionIDIndex in mentions];
        if len(mentions) > 0 and len(mentions) <= _cfg['max_size']:
            #print _p_.memory_info().rss/_mb_, 'MB used';
            D = DBSCAN(D,mentions) if _dbscan_ else AGGLO(D,mentions);
            #D = DBSCAN_SKLEARN(D,mentions);
            #D = AGGLO(D,mentions);
    D.MC      = csr((np.ones(len(D.labelling),bool),(np.arange(len(D.labelling)),D.labelling)),shape=(len(D.labelling),len(D.labels)),dtype=bool);
    D.NC      = D.NM.dot(D.MC);
    D.rids_c  = D.MC.T.dot(D.MR_);
    D.new     = np.zeros(D.new.shape,dtype=bool);
    TIME[CLU] = time.time() - t_clu; CLUS = len(new_nodes);
    return D;

def merge(D,group):
    global MEM;
    #r        = group[np.argmin([len(D.nodes[D.index2node[i]][REP]) for i in group])];                                                                               mem=_p_.memory_info().rss/_mb_; MEM[MER]=max(MEM[MER],mem);
    r        = group[np.argmax([D.NM[i,:].sum() for i in group])]; print 'merging', [D.index2node[i] for i in group], 'into', D.index2node[r];                      mem=_p_.memory_info().rss/_mb_; MEM[MER]=max(MEM[MER],mem);
    remove   = [x for i,x in enumerate(group) if x!=r];                                                                                                             mem=_p_.memory_info().rss/_mb_; MEM[MER]=max(MEM[MER],mem);
    keep     = sorted(list(set(range(D.spec.shape[0]))-set(remove)));  #print remove, keep;                                                                          mem=_p_.memory_info().rss/_mb_; MEM[MER]=max(MEM[MER],mem);
    D.edge   = combine(D.edge,group,r,keep,False);                                                                                                                  mem=_p_.memory_info().rss/_mb_; MEM[MER]=max(MEM[MER],mem);
    old_size = D.NM[r,:].sum(); print '$$$ OLD SIZE:',old_size,'$$$';
    D.NM     = combine(D.NM,group,r,keep,False);                                                                                                                    mem=_p_.memory_info().rss/_mb_; MEM[MER]=max(MEM[MER],mem);
    new_size = D.NM[keep.index(r),:].sum(); print '$$$ NEW SIZE:',new_size,'$$$';
    D.rids_b = D.NM.dot(D.MR_);                                                                                                                                     mem=_p_.memory_info().rss/_mb_; MEM[MER]=max(MEM[MER],mem);
    D.spec   = combine(D.spec,group,r,keep,True);                                                                                                                   mem=_p_.memory_info().rss/_mb_; MEM[MER]=max(MEM[MER],mem);
    D.obs    = combine(D.obs,group,r,keep,False);                                                                                                                   mem=_p_.memory_info().rss/_mb_; MEM[MER]=max(MEM[MER],mem);
    D.car    = (D.obs.T.dot(D.spec.T)).T;                                                                                                                           mem=_p_.memory_info().rss/_mb_; MEM[MER]=max(MEM[MER],mem);
    D.update_weights();
    D.nodes[D.index2node[r]][RID] = sum([D.nodes[D.index2node[i]][RID] for i in group],Counter()); #TODO: Check if still required
    mem=_p_.memory_info().rss/_mb_; MEM[MER]=max(MEM[MER],mem);
    for i in remove: D.nodes[D.index2node[i]][RID] = Counter();                                    #TODO: Check if still required
    mem=_p_.memory_info().rss/_mb_; MEM[MER]=max(MEM[MER],mem);
    D.update_index(keep,r,old_size!=new_size);                                                                                                                                         mem=_p_.memory_info().rss/_mb_; MEM[MER]=max(MEM[MER],mem);
    return D;

def get_groups_(D,t):
    if _random_:
        if _nbrdm_:
            to_merge = D.edge; 
        else:
            labels   = connected_components(D.edge)[1];
            to_merge = (labels[:,None]==labels);
    else:
        nm       = np.array(D.NM.sum(1)).astype(float);
        p_new    = np.nan_to_num(1-((nm**2+nm.T**2)/((nm+nm.T)**2))) if _p_new_ else np.full(D.weight.shape,0.5); np.fill_diagonal(p_new,0.0);
        score    = D.weight.multiply(p_new*2); #The *2 is only such that the optimum of two equal sized blocks gets 1-weighting
        kth      = np.argsort(score.data)[-min(score.data.size,_top_k_)] if _top_k_ != None else None;
        t        = max(0.0000000001,score.data[kth]) if _top_k_ != None else t;
        #if _top_k_ != None: print score.data[kth], '=', np.ravel(p_new)[kth], '(p_new) x', D.weight.data[kth], '(weight)';
        to_merge = score >= t;
    labels    = connected_components(to_merge)[1];
    sorting   = np.argsort(labels);
    labels_s  = labels[sorting];
    _, starts = np.unique(labels_s,return_index=True);
    sizes     = np.diff(starts);
    groups    = [group for group in np.split(sorting,starts[1:]) if group.size > 1];
    if _random_ != None:
        groups = np.random.permutation(groups)[:min(len(groups),_cfg['num_rdm'])];
    return groups;

def get_groups(D,t):
    nm = np.array(D.NM.sum(1)).astype(float);
    if _random_:
        ok_size = nm+nm.T <= _cfg['max_size'];
        edges   = D.edge.multiply(ok_size).toarray();
        np.fill_diagonal(edges,False);
        if _nbrdm_:
            if _p_new_:
                p_new = 1-((nm**2+nm.T**2)/((nm+nm.T)**2)); p_new[np.isnan(p_new)]=0.5; p_new[p_new==0]=0.5; np.fill_diagonal(p_new,0.0);
                rows,cols = np.argwhere(p_new==np.amax(p_new)).T;
                #for i in xrange(len(rows)):
                #    print D.index2node[rows[i]],'-->',D.index2node[cols[i]];
            else:
                rows,cols = edges.nonzero();
                #for i in xrange(len(rows)):
                #    print D.index2node[rows[i]],'-->',D.index2node[cols[i]];
        else:
            labels    = connected_components(edges)[1];
            rows,cols = np.triu(labels[:,None]==labels).nonzero();
        selects     = np.random.choice(range(len(rows)),min(len(rows),_cfg['num_rdm']),replace=False) if len(rows)>0 else [];
        rows_,cols_ = rows[selects], cols[selects];
        to_merge    = csr((np.ones(len(rows_),bool),(rows_,cols_)),shape=edges.shape);
    else:#TODO:check the below for correctness
        p_new    = 1-((nm**2+nm.T**2)/((nm+nm.T)**2)) if _p_new_ else np.full(D.weight.shape,0.5); p_new[np.isnan(p_new)]=0.5; p_new[p_new==0]=0.5; np.fill_diagonal(p_new,0.0);
        #print D.NM; print nm; print p_new; print D.weight;
        score    = D.weight.multiply(p_new*2); #The *2 is only such that the optimum of two equal sized blocks gets 1-weighting
        #print score;
        kth      = np.argsort(score.data)[-min(score.data.size,_top_k_)] if _top_k_ != None else None;
        t        = max(0.0000000001,score.data[kth]) if _top_k_ != None else t;
        #if _top_k_ != None: print score.data[kth], '=', np.ravel(p_new)[kth], '(p_new) x', D.weight.data[kth], '(weight)';
        to_merge = score >= t;
    labels    = connected_components(to_merge)[1];
    sorting   = np.argsort(labels);
    labels_s  = labels[sorting];
    _, starts = np.unique(labels_s,return_index=True);
    sizes     = np.diff(starts);
    groups    = [group for group in np.split(sorting,starts[1:]) if group.size > 1];
    return groups;

def merger(D,t):
    global TIME, MERGE, BOUND;
    mer_t     = time.time();
    groups    = get_groups(D,t);
    groups    = [[D.index2node[i] for i in group] for group in groups];  #TODO: Keep the node names, because the indices are changing!
    for group in groups:
        group_idx = [D.node2index[name] for name in group];
        if D.NM[group_idx,:].sum() <= _cfg['max_size']:
            D = merge(D,group_idx); #print _p_.memory_info().rss/_mb_, 'MB used';
        else:
            BOUND += 1;
            print 'group union too large!';
    TIME[MER] = time.time() - mer_t; MERGE = len(groups);
    return D;

def discounter(D,d):
    global TIME, MEM;
    dis_t    = time.time();
    O_dense  = D.obs.toarray();                                                                         mem=_p_.memory_info().rss/_mb_; MEM[DIS]=max(MEM[DIS],mem); #print mem,'MB used';
    distrib  = D.weight.multiply(csr(1./D.weight.sum(1),shape=(D.weight.shape[0],1),dtype=float));      mem=_p_.memory_info().rss/_mb_; MEM[DIS]=max(MEM[DIS],mem); #print mem,'MB used';
    discount = D.obs*d;                                                                                 mem=_p_.memory_info().rss/_mb_; MEM[DIS]=max(MEM[DIS],mem); #print mem,'MB used';
    gain     = (discount.T.dot(distrib)).T;                                                             mem=_p_.memory_info().rss/_mb_; MEM[DIS]=max(MEM[DIS],mem); #print mem,'MB used';
    O_dense -= discount.toarray();                                                                      mem=_p_.memory_info().rss/_mb_; MEM[DIS]=max(MEM[DIS],mem); #print mem,'MB used';
    O_dense += gain.toarray();                                                                          mem=_p_.memory_info().rss/_mb_; MEM[DIS]=max(MEM[DIS],mem); #print mem,'MB used';
    D.obs    = csr(O_dense,shape=D.obs.shape,dtype=float);                                              mem=_p_.memory_info().rss/_mb_; MEM[DIS]=max(MEM[DIS],mem); #print mem,'MB used';
    D.car    = (D.obs.T.dot(D.spec.T)).T;                                                               mem=_p_.memory_info().rss/_mb_; MEM[DIS]=max(MEM[DIS],mem); #print mem,'MB used';
    D.update_weights();
    TIME[DIS] = time.time() - dis_t;
    return D;
#-------------------------------------------------------------------------------------------------------------------------------------
#-FUNCTIONS-INTERFACE-----------------------------------------------------------------------------------------------------------------

def samples(D,same_file,diff_file):
    same_rid = [D.MR_[:,1:][:,i].nonzero()[0] for i in xrange(D.MR_[:,1:].shape[1])];
    num_same = sum([(len(el)**2) for el in same_rid]);
    diff_rid = [];
    num_diff = 0;
    for i in xrange(len(same_rid)):
        if num_diff <= num_same and (same_rid[:i]!=[] or same_rid[i+1:]!=[]):
            diff_rid.append([same_rid[i],np.concatenate(same_rid[:i]+same_rid[i+1:])]);
            num_diff += len(diff_rid[-1][0])*len(diff_rid[-1][0]);
    tmp_sames = [sim(same_rid[i],D,same_rid[i]) for i in xrange(len(same_rid))];
    tmp_sames = [tmp_sames[i][(np.triu(tmp_sames[i],1)+np.tril(tmp_sames[i],-1)).nonzero()] for i in xrange(len(same_rid))];
    tmp_diffs = [np.ravel(sim(diff_rid[i][0],D,diff_rid[i][1])) for i in xrange(len(diff_rid))];
    similarities_same = np.concatenate( tmp_sames );
    similarities_diff = np.concatenate( tmp_diffs )[:len(similarities_same)];
    print similarities_same; print similarities_diff;
    OUT=open(same_file,'a'); OUT.write('\n'.join([str(similarities_same[i]) for i in xrange(len(similarities_same))])+'\n'); OUT.close();
    OUT=open(diff_file,'a'); OUT.write('\n'.join([str(similarities_diff[i]) for i in xrange(len(similarities_diff))])+'\n'); OUT.close();

def progress(D,t_start,con_out,cur_out):
    global iteration, COMP;
    base_prec, base_rec, base_f1 = prec_rec_f1(csr(D.MR_[:,1:].sum(0))); print 'basePrec:', base_prec, 'baseRec:', base_rec, 'base_f1:', base_f1;
    I = 0; B = _surname+', '+_firstinit if _firstinit!=None else _surname; COMP = np.zeros(D.NM.shape[1],dtype=int); c_time_0 = time.clock();
    D = clusterer(D); #print '---------------------------------------------------------------done clustering.';
    thr_iter = _cfg['thr'];
    output(D,I+1,B,thr_iter,t_start,0,time.clock()-c_time_0,con_out,cur_out);
    while thr_iter >= 0:#(_cfg['thr']+mvl)-(I*_cfg['step']) >= mvl:
        #draw(D,colors,I);
        #thr_iter = (_cfg['thr']+mvl)-(I*_cfg['step']);
        m_time_0 = time.clock();print 't =',thr_iter; print len(D.index2node);
        D      = merger(D,thr_iter);         #print '---------------------------------------------------------------done merging.';
        #draw(D,colors,I);
        D      = discounter(D,_d_);          #print '---------------------------------------------------------------done discounting.';
        #draw(D,colors,I);
        m_time = time.clock() - m_time_0; c_time_0 = time.clock();
        D      = clusterer(D);               #print '---------------------------------------------------------------done clustering.';
        c_time = time.clock() - c_time_0;
        I += 1;
        thr_iter -= _cfg['step'];
        output(D,I+1,B,thr_iter,t_start,m_time,c_time,con_out,cur_out);

def interface(D,colors):
    global iteration, COMP;
    old_D = copy(D);          old_iteration = iteration; I = 0; B = _cfg['root_dir']+_cfg['name_db'];
    mvl   = 0.00000000000001; 
    COMP  = np.zeros(D.NM.shape[1],dtype=int);
    while (_cfg['thr']+mvl)-(I*_cfg['step']) >= mvl:
        sanity_check(D);
        draw(D,colors,0); print 't =',(_cfg['thr']+mvl)-(I*_cfg['step']); print len(D.index2node);
        option=raw_input("... m(erge) - d(iscount) - c(luster) - r(eset) ...");
        if option=='m':   #-MERGE------------------------------------------------------------------------------------------------------------
            old_D = copy(D); old_iteration = iteration;
            D     = merger(D,(_cfg['thr']+mvl)-(I*_cfg['step']));    print '---------------------------------------------------------------done merging.';
            I    += 1;
        elif option=='d': #-DISCOUNT---------------------------------------------------------------------------------------------------------
            old_D = copy(D); old_iteration = iteration;
            D     = discounter(D,1.0);              print '---------------------------------------------------------------done discounting.';
        elif option=='c': #-CLUSTER----------------------------------------------------------------------------------------------------------
            old_D = copy(D); old_iteration = iteration;
            D     = clusterer(D);                   print '---------------------------------------------------------------done clustering.';
        elif option=='r': #-RESET------------------------------------------------------------------------------------------------------------
            D = old_D; iteration = old_iteration;   print '---------------------------------------------------------------done resetting.';
        else:
            print 'No such option.';
#-------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#SCRIPT---------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------
if _cfg['mode'] != 'sample':
    con_out = sqlite3.connect(_result_db);
    cur_out = con_out.cursor();
    if _checker_:
        new_db = cur_out.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='results'").fetchall() == []; print 'new_db:', new_db;
        if not new_db:
            num_rows = cur_out.execute("SELECT count(*) FROM results").fetchall()[0][0]; print 'num_rows:', num_rows;
            if num_rows==402: exit('This has already been calculated. Skipping...');
            cur_out.execute("DROP TABLE IF EXISTS results"); cur_out.execute("DROP TABLE IF EXISTS tuning");
    cur_out.execute("CREATE TABLE IF NOT EXISTS results(t_start TEXT, t_iter TEXT, bottom TEXT, eps REAL, z REAL, r REAL, minPts INT, smooth REAL, iteration INTEGER, num_m INTEGER, num_r INTEGER, merge INT, clus INT, comp REAL, blocksum INT, bound INT, num_b INTEGER, pre_b REAL, rec_b REAL, f1_b REAL, tp_b REAL, t_b REAL, p_b REAL, num_c INTEGER, pre_c REAL, rec_c REAL, f1_c REAL, tp_c REAL, t_c REAL, p_c REAL, max_p REAL, max_r REAL, max_f1 REAL, max_size INT, mem_clu INT, mem_mer INT, mem_dis INT, time_clu REAL, time_sim REAL, time_alg REAL, time_mer REAL, time_dis REAL, cpu_m REAL, cpu_c REAL)");
    cur_out.execute("CREATE TABLE IF NOT EXISTS tuning(stamp TEXT, min_sim REAL, max_f1 REAL, size INT, left INT, right INT)");
#-------------------------------------------------------------------------------------------------------------------------------------
print 'Building graph...'; t = time.time();
#-------------------------------------------------------------------------------------------------------------------------------------
lat_els, lat_edg = load_lattice(_cfg['lat_file']);
node_infos       = load_node_infos_db(_cfg['root_dir']+_cfg['name_db'],_surname,_firstinit);
#node_infos       = node_infos if not _cfg['milojevic'] else [[set([tup for tup in set_rep if not tup[0] in ['f1_','f2_','f3_']]),rIDcounts] for set_rep,rIDcounts in node_infos];#TODO: remove this line
#-------------------------------------------------------------------------------------------------------------------------------------
print time.time()-t, 'seconds for loading data.'; t = time.time();
#-------------------------------------------------------------------------------------------------------------------------------------
observed_nodes = [make_node(node_info,_cfg['aggregate']) for node_info in node_infos if node_info[1].keys()!=[None] or not _cfg['only_rIDs']];
observed_nodes = [observed_node for observed_node in observed_nodes if observed_node[TYP] in lat_els];
mentions       = [(node[REP],rID,1.0,None,) for node in observed_nodes for rID in node[RID] for i in xrange(int(node[RID][rID]))] if _cfg['aggregate'] else [(node[REP],rID,1.0,mentionID,) for node in observed_nodes for rID in node[RID] for mentionID in node[RID][rID]];
rIDs           = sorted(list(set([rID for i in xrange(len(observed_nodes)) for rID in observed_nodes[i][RID]])));
#-------------------------------------------------------------------------------------------------------------------------------------
print 'Number of rIDs:', len(rIDs);
if len(rIDs) == 0: exit();
print 'First rID is', rIDs[0];#sort so that None is up front
#-------------------------------------------------------------------------------------------------------------------------------------
colorscheme    = get_colors(len(rIDs)+0)[0:];
colors         = {i:colorscheme[i] for i in xrange(len(rIDs))}; colors[0] = (0.,0.,1.) if rIDs[0]==None else colors[0]; #white is for None
Nodes          = dict();
iteration      = 0;
#-------------------------------------------------------------------------------------------------------------------------------------
print time.time()-t, 'seconds for preprocessing.'; t = time.time();
#-------------------------------------------------------------------------------------------------------------------------------------
for i in xrange(len(observed_nodes)):
    if observed_nodes[i][RID].keys() != []:
        add_node(observed_nodes[i][REP],observed_nodes[i][RID],_cfg['aggregate'],Nodes);
#-------------------------------------------------------------------------------------------------------------------------------------
print time.time()-t, 'seconds for adding nodes.'; t = time.time();
#-------------------------------------------------------------------------------------------------------------------------------------
D = DATA(Nodes,mentions,rIDs,lat_els,lat_edg,False,_cfg['aggregate']);
#-------------------------------------------------------------------------------------------------------------------------------------
print time.time()-t, 'seconds for making representation.'; t = time.time();
#-------------------------------------------------------------------------------------------------------------------------------------
node_sizes = sorted([(D.NM[i,:].sum(1),D.index2node[i]) for i in xrange(D.NM.shape[0])]);
for size, node_str in node_sizes:
    if size > 0: print node_str, size;
#-------------------------------------------------------------------------------------------------------------------------------------
D_old   = copy(D);
t_start = datetime.datetime.utcnow().isoformat().replace('T',' ')[:-7];
if _cfg['mode'] == 'interface':
    interface(D,colors);
elif _cfg['mode'] == 'sample':
    samples(D,''.join(_result_db.split('.')[:-1])+'_'+_cfg['same_file'],''.join(_result_db.split('.')[:-1])+'_'+_cfg['diff_file']);
else:                                    
    progress(D,t_start,con_out,cur_out);
D = copy(D_old);
# prob_for('terms',np.array(range(D.NM.shape[1])),D).sum(0) for testing normalization
#-------------------------------------------------------------------------------------------------------------------------------------
print time.time()-t, 'seconds for main processing.'; t = time.time();
#-------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
