#ifndef _IMU_PROCESSING_HPP
#define _IMU_PROCESSING_HPP

#include <cmath>
#include <math.h>
#include <deque>
#include <mutex>
#include <thread>
#include <fstream>
#include <csignal>
#include <ros/ros.h>
#include <so3_math.h>
#include <Eigen/Eigen>
#include <common_lib.h>
#include <pcl/common/io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <condition_variable>
#include <nav_msgs/Odometry.h>
#include <pcl/common/transforms.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <tf/transform_broadcaster.h>
#include <eigen_conversions/eigen_msg.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <lidar_imu_init/States.h>
#include <geometry_msgs/Vector3.h>

/// *************Preconfiguration

#define MAX_INI_COUNT (200)

const bool time_list(PointType &x, PointType &y) {return (x.curvature < y.curvature);};

/// *************IMU Process and undistortion
class ImuProcess
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ImuProcess();
  ~ImuProcess();
  
  void Reset();
  void Reset(double start_timestamp, const sensor_msgs::ImuConstPtr &lastimu);
  void set_R_LI_cov(const V3D &R_LI_cov);
  void set_T_LI_cov(const V3D &T_LI_cov);
  void set_gyr_cov(const V3D &scaler);
  void set_acc_cov(const V3D &scaler);
  void set_mean_acc_norm(const double &mean_acc_norm);
  void set_gyr_bias_cov(const V3D &b_g);
  void set_acc_bias_cov(const V3D &b_a);
  void Process(const MeasureGroup &meas, StatesGroup &state, PointCloudXYZI::Ptr pcl_un_);


//  ros::NodeHandle nh;
  ofstream fout_imu;
  V3D cov_acc;
  V3D cov_gyr;
  V3D cov_R_LI;
  V3D cov_T_LI;
  V3D cov_acc_scale;
  V3D cov_gyr_scale;
  V3D cov_bias_gyr;
  V3D cov_bias_acc;
  double first_lidar_time;
  int    lidar_type;
  bool   imu_en;
  bool LI_init_done = false;
  double IMU_mean_acc_norm;

 private:
  void IMU_init(const MeasureGroup &meas, StatesGroup &state, int &N);
  void propagation_and_undist(const MeasureGroup &meas, StatesGroup &state_inout, PointCloudXYZI &pcl_in_out);
  void Forward_propagation_without_imu(const MeasureGroup &meas, StatesGroup &state_inout, PointCloudXYZI &pcl_out);
  PointCloudXYZI::Ptr cur_pcl_un_;
  sensor_msgs::ImuConstPtr last_imu_;
  deque<sensor_msgs::ImuConstPtr> v_imu_;
  vector<Pose6D> IMUpose;
  V3D mean_acc;
  V3D mean_gyr;
  V3D angvel_last;
  V3D acc_s_last;
  double last_lidar_end_time_;
  double time_last_scan;
  int    init_iter_num = 1;
  bool   b_first_frame_ = true;
  bool   imu_need_init_ = true;
};

ImuProcess::ImuProcess()
    : b_first_frame_(true), imu_need_init_(true)
{
  imu_en = true;
  init_iter_num = 1;
  cov_acc         = V3D(0.1, 0.1, 0.1);
  cov_gyr         = V3D(0.1, 0.1, 0.1);
  cov_R_LI        = V3D(0.00001, 0.00001, 0.00001);
  cov_T_LI        = V3D(0.0001, 0.0001, 0.0001);
  cov_bias_gyr    = V3D(0.0001, 0.0001, 0.0001);
  cov_bias_acc    = V3D(0.0001, 0.0001, 0.0001);
  mean_acc        = V3D(0, 0, -1.0);
  mean_gyr        = V3D(0, 0, 0);
  angvel_last     = Zero3d;
  last_imu_.reset(new sensor_msgs::Imu());
  fout_imu.open(DEBUG_FILE_DIR("imu.txt"),ios::out);
}

ImuProcess::~ImuProcess() {}

void ImuProcess::Reset() 
{
  ROS_WARN("Reset ImuProcess");
  mean_acc      = V3D(0, 0, -1.0);
  mean_gyr      = V3D(0, 0, 0);
  angvel_last       = Zero3d;
  imu_need_init_    = true;
  init_iter_num     = 1;
  v_imu_.clear();
  IMUpose.clear();
  last_imu_.reset(new sensor_msgs::Imu());
  cur_pcl_un_.reset(new PointCloudXYZI());
}


void ImuProcess::set_gyr_cov(const V3D &scaler)
{
  cov_gyr_scale = scaler;
}

void ImuProcess::set_acc_cov(const V3D &scaler)
{
  cov_acc_scale = scaler;
}

void ImuProcess::set_R_LI_cov(const V3D &R_LI_cov)
{
    cov_R_LI = R_LI_cov;
}

void ImuProcess::set_T_LI_cov(const V3D &T_LI_cov)
{
    cov_T_LI = T_LI_cov;
}

void ImuProcess::set_mean_acc_norm(const double &mean_acc_norm){
    IMU_mean_acc_norm = mean_acc_norm;
}

void ImuProcess::set_gyr_bias_cov(const V3D &b_g)
{
  cov_bias_gyr = b_g;
}

void ImuProcess::set_acc_bias_cov(const V3D &b_a)
{
  cov_bias_acc = b_a;
}

void ImuProcess::IMU_init(const MeasureGroup &meas, StatesGroup &state_inout, int &N)
{
  /** 1. initializing the gravity, gyro bias, acc and gyro covariance
   ** 2. normalize the acceleration measurements to unit gravity **/
  //初始化重力、陀螺仪偏差、acc和陀螺仪协方差  将加速度测量值归一化为单位重力
  ROS_INFO("IMU Initializing: %.1f %%", double(N) / MAX_INI_COUNT * 100);
  V3D cur_acc, cur_gyr;
  
  if (b_first_frame_)
  {
    Reset();  //重置IMU参数
    N = 1;    //将迭代次数置1
    b_first_frame_ = false;
    const auto &imu_acc = meas.imu.front()->linear_acceleration;   //IMU初始时刻的加速度
    const auto &gyr_acc = meas.imu.front()->angular_velocity;     //IMU初始时刻的角速度
    mean_acc << imu_acc.x, imu_acc.y, imu_acc.z;                //第一帧加速度值作为初始化均值
    mean_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;                //第一帧角速度值作为初始化均值
    first_lidar_time = meas.lidar_beg_time;                   //将当前IMU帧对应的lidar起始时间 作为初始时间
  }

  for (const auto &imu : meas.imu)    //根据所有IMU数据，计算平均值和方差
  {
    const auto &imu_acc = imu->linear_acceleration;
    const auto &gyr_acc = imu->angular_velocity;
    cur_acc << imu_acc.x, imu_acc.y, imu_acc.z;
    cur_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;

    mean_acc      += (cur_acc - mean_acc) / N;    //根据当前帧和均值差作为均值的更新
    mean_gyr      += (cur_gyr - mean_gyr) / N;

    cov_acc = cov_acc * (N - 1.0) / N + (cur_acc - mean_acc).cwiseProduct(cur_acc - mean_acc) * (N - 1.0) / (N * N);
    cov_gyr = cov_gyr * (N - 1.0) / N + (cur_gyr - mean_gyr).cwiseProduct(cur_gyr - mean_gyr) * (N - 1.0) / (N * N);

    N ++;
  }

  state_inout.gravity = - mean_acc / mean_acc.norm() * G_m_s2;    //得平均测量的单位方向向量 * 重力加速度预设值
  state_inout.rot_end = Eye3d;    //初始旋转为I
  state_inout.bias_g.setZero();   //初始bg为0
  last_imu_ = meas.imu.back();
}



void ImuProcess::Forward_propagation_without_imu(const MeasureGroup &meas, StatesGroup &state_inout,
                             PointCloudXYZI &pcl_out) {
    pcl_out = *(meas.lidar);
    /*** sort point clouds by offset time ***/
    const double &pcl_beg_time = meas.lidar_beg_time;
    sort(pcl_out.points.begin(), pcl_out.points.end(), time_list);   // 根据点云中每个点的时间戳对点云进行重排序
    const double &pcl_end_offset_time = pcl_out.points.back().curvature / double(1000);  //这里curvature中存放了时间戳（在preprocess.cpp中）

    MD(DIM_STATE, DIM_STATE) F_x, cov_w;   //定义18*18的矩阵fx 和cov_w=FW * Q q* FW^T  对应公式(5)fx  注意这里的状态量有6个(R P V g ba bg) 和论文中4个不一样
    double dt = 0.0;

    if (b_first_frame_) {  //初始为true
        dt = 0.1;
        b_first_frame_ = false;
    } else {
        dt = pcl_beg_time - time_last_scan;
        time_last_scan = pcl_beg_time;
    }

    /* covariance propagation */
    F_x.setIdentity();
    cov_w.setZero();
    /** In CV model, bias_g represents angular velocity **/       //CV模型中 bg对应角速度
    /** In CV model，bias_a represents linear acceleration **/    //CV模型中 ba对应加速度
    M3D Exp_f = Exp(state_inout.bias_g, dt);     
    F_x.block<3, 3>(0, 0) = Exp(state_inout.bias_g, -dt);    //对应公式(5)第1行第1列
    F_x.block<3, 3>(0, 15) = Eye3d * dt;    //对应公式(5)第1行第4列
    F_x.block<3, 3>(3, 12) = Eye3d * dt;    //对应公式(5)第2行第3列


    cov_w.block<3, 3>(15, 15).diagonal() = cov_gyr_scale * dt * dt;    //定义18*18的矩阵fx 和cov_w=FW * Q * FW^T 
    cov_w.block<3, 3>(12, 12).diagonal() = cov_acc_scale * dt * dt;

    /** Forward propagation of covariance**/
    state_inout.cov = F_x * state_inout.cov * F_x.transpose() + cov_w;  //对应公式(4)

    /** Forward propagation of attitude **/
    state_inout.rot_end = state_inout.rot_end * Exp_f;  //对应公式(3)旋转部分

    /** Position Propagation **/
    state_inout.pos_end += state_inout.vel_end * dt;  //对应公式(3)平移部分

    /**CV model： un-distort pcl using linear interpolation **/
    if(lidar_type != L515){
        auto it_pcl = pcl_out.points.end() - 1;
        double dt_j = 0.0;
        for(; it_pcl != pcl_out.points.begin(); it_pcl --)
        {
            dt_j= pcl_end_offset_time - it_pcl->curvature/double(1000);
            M3D R_jk(Exp(state_inout.bias_g, - dt_j));
            V3D P_j(it_pcl->x, it_pcl->y, it_pcl->z);
            // Using rotation and translation to un-distort points
            V3D p_jk;
            p_jk = - state_inout.rot_end.transpose() * state_inout.vel_end * dt_j;

            V3D P_compensate =  R_jk * P_j + p_jk;

            /// save Undistorted points and their rotation
            it_pcl->x = P_compensate(0);
            it_pcl->y = P_compensate(1);
            it_pcl->z = P_compensate(2);
        }
    }
}

void ImuProcess::propagation_and_undist(const MeasureGroup &meas, StatesGroup &state_inout, PointCloudXYZI &pcl_out)
{
  /*** add the imu of the last frame-tail to the current frame-head ***/
  pcl_out = *(meas.lidar);
  auto v_imu = meas.imu;          //取出当前帧的IMU队列
  v_imu.push_front(last_imu_);    //将上一帧最后尾部的imu添加到当前帧头部的imu
  double imu_end_time = v_imu.back()->header.stamp.toSec();     //拿到当前帧尾部的imu的时间
  double pcl_beg_time, pcl_end_time;

  if (lidar_type == L515)
  {
    pcl_beg_time = last_lidar_end_time_;
    pcl_end_time = meas.lidar_beg_time;
  }
  else
  {
    pcl_beg_time = meas.lidar_beg_time;
    /*** sort point clouds by offset time ***/
    sort(pcl_out.points.begin(), pcl_out.points.end(), time_list);   // 根据点云中每个点的时间戳对点云进行重排序
    pcl_end_time = pcl_beg_time + pcl_out.points.back().curvature / double(1000);
  }


  /*** Initialize IMU pose ***/
  IMUpose.clear();
  IMUpose.push_back(set_pose6d(0.0, acc_s_last, angvel_last, state_inout.vel_end, state_inout.pos_end, state_inout.rot_end));
  //将初始状态加入IMUpose中,包含有时间间隔，上一帧加速度，上一帧角速度，上一帧速度，上一帧位置，上一帧旋转矩阵

  /*** forward propagation at each imu point ***/
  V3D acc_imu, angvel_avr, acc_avr, vel_imu(state_inout.vel_end), pos_imu(state_inout.pos_end);   
  M3D R_imu(state_inout.rot_end);
  MD(DIM_STATE, DIM_STATE) F_x, cov_w;
  
  double dt = 0;
  for (auto it_imu = v_imu.begin(); it_imu < (v_imu.end() - 1); it_imu++)
  {
    auto &&head = *(it_imu);
    auto &&tail = *(it_imu + 1);

    if (tail->header.stamp.toSec() < last_lidar_end_time_)    continue;    //判断时间先后顺序：下一帧时间戳是否小于上一帧结束时间戳 不符合直接continue
    
    angvel_avr<<0.5 * (head->angular_velocity.x + tail->angular_velocity.x),     // 中值积分
                0.5 * (head->angular_velocity.y + tail->angular_velocity.y),
                0.5 * (head->angular_velocity.z + tail->angular_velocity.z);


    acc_avr   <<0.5 * (head->linear_acceleration.x + tail->linear_acceleration.x),
                0.5 * (head->linear_acceleration.y + tail->linear_acceleration.y),
                0.5 * (head->linear_acceleration.z + tail->linear_acceleration.z);

      V3D angvel_now(head->angular_velocity.x, head->angular_velocity.y, head->angular_velocity.z);
      V3D acc_now(head->linear_acceleration.x, head->linear_acceleration.y, head->linear_acceleration.z);
      fout_imu << setw(10) << head->header.stamp.toSec() << "  " << angvel_now.transpose()<< " " << acc_now.transpose() << endl;

    angvel_avr -= state_inout.bias_g;         //角速度-bias
    acc_avr     = acc_avr / IMU_mean_acc_norm * G_m_s2 - state_inout.bias_a;   //加速度尺度校正 - bias

    if(head->header.stamp.toSec() < last_lidar_end_time_)  //如果IMU开始时刻早于上次雷达最晚时刻(因为将上次最后一个IMU插入到此次开头了，所以会出现一次这种情况)
        dt = tail->header.stamp.toSec() - last_lidar_end_time_;  //从上次雷达时刻末尾开始传播 计算与此次IMU结尾之间的时间差
    else
        dt = tail->header.stamp.toSec() - head->header.stamp.toSec();  //两个IMU时刻之间的时间间隔
    
    /* covariance propagation */
    M3D acc_avr_skew;
    M3D Exp_f   = Exp(angvel_avr, dt);
    acc_avr_skew<<SKEW_SYM_MATRX(acc_avr);

    // 以下为FAST-LIO前向传播公式
    F_x.setIdentity();
    cov_w.setZero();

    F_x.block<3,3>(0,0)  = Exp(angvel_avr, -dt);
    F_x.block<3,3>(0,15)  = - Eye3d * dt;
    F_x.block<3,3>(3,12)  = Eye3d * dt;
    F_x.block<3,3>(12,0)  = - R_imu * acc_avr_skew * dt;
    F_x.block<3,3>(12,18) = - R_imu * dt;
    F_x.block<3,3>(12,21) = Eye3d * dt;

    cov_w.block<3,3>(0,0).diagonal()   = cov_gyr * dt * dt;
    cov_w.block<3,3>(6,6).diagonal()   = cov_R_LI * dt * dt;
    cov_w.block<3,3>(9,9).diagonal()   = cov_T_LI * dt * dt;
    cov_w.block<3,3>(12,12)            = R_imu * cov_acc.asDiagonal() * R_imu.transpose() * dt * dt;
    cov_w.block<3,3>(15,15).diagonal() = cov_bias_gyr * dt * dt; // bias gyro covariance
    cov_w.block<3,3>(18,18).diagonal() = cov_bias_acc * dt * dt; // bias acc covariance

    state_inout.cov = F_x * state_inout.cov * F_x.transpose() + cov_w;

    /* propagation of IMU attitude (global frame)*/
    R_imu = R_imu * Exp_f;

    /* Specific acceleration (global frame) of IMU */
    acc_imu = R_imu * acc_avr + state_inout.gravity;

    /* propagation of IMU position (global frame)*/
    pos_imu = pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt;

    /* velocity of IMU (global frame)*/
    vel_imu = vel_imu + acc_imu * dt;

    /* save the poses at each IMU measurements (global frame)*/
    angvel_last = angvel_avr;
    acc_s_last  = acc_imu;
    double &&offs_t = tail->header.stamp.toSec() - pcl_beg_time;
    IMUpose.push_back(set_pose6d(offs_t, acc_imu, angvel_avr, vel_imu, pos_imu, R_imu));
  }

  /*** calculated the pos and attitude prediction at the frame-end ***/
  double note = pcl_end_time > imu_end_time ? 1.0 : -1.0;
  dt = note * (pcl_end_time - imu_end_time);
  state_inout.vel_end = vel_imu + note * acc_imu * dt;
  state_inout.rot_end = R_imu * Exp(V3D(note * angvel_avr), dt);
  state_inout.pos_end = pos_imu + note * vel_imu * dt + note * 0.5 * acc_imu * dt * dt;


  last_imu_ = meas.imu.back();
  last_lidar_end_time_ = pcl_end_time;

  if (lidar_type != L515)
  {
    #ifdef DEBUG_PRINT
      cout<<"[ IMU Process ]: vel "<<state_inout.vel_end.transpose()<<" pos "<<state_inout.pos_end.transpose()<<" ba"<<state_inout.bias_a.transpose()<<" bg "<<state_inout.bias_g.transpose()<<endl;
      cout<<"propagated cov: "<<state_inout.cov.diagonal().transpose()<<endl;
    #endif
    /*** un-distort each lidar point (backward propagation) ***/
    auto it_pcl = pcl_out.points.end() - 1; //a single point in k-th frame
    for (auto it_kp = IMUpose.end() - 1; it_kp != IMUpose.begin(); it_kp--)
    {
        auto head = it_kp - 1;
        R_imu << MAT_FROM_ARRAY(head->rot);
        acc_imu << VEC_FROM_ARRAY(head->acc);
        // cout<<"head imu acc: "<<acc_imu.transpose()<<endl;
        vel_imu << VEC_FROM_ARRAY(head->vel);
        pos_imu << VEC_FROM_ARRAY(head->pos);
        angvel_avr << VEC_FROM_ARRAY(head->gyr);
        for (; it_pcl->curvature / double(1000) > head->offset_time; it_pcl--) {
            dt = it_pcl->curvature / double(1000) - head->offset_time; //dt = t_j - t_i > 0
            /* Transform to the 'scan-end' IMU frame（I_k frame)*/
            M3D R_i(R_imu * Exp(angvel_avr, dt));
            V3D P_i = pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt;
            V3D p_in(it_pcl->x, it_pcl->y, it_pcl->z);
            V3D P_compensate = state_inout.offset_R_L_I.transpose() * (state_inout.rot_end.transpose() * (R_i * (state_inout.offset_R_L_I * p_in + state_inout.offset_T_L_I) + P_i - state_inout.pos_end) - state_inout.offset_T_L_I);
            /// save Undistorted points
            it_pcl->x = P_compensate(0);
            it_pcl->y = P_compensate(1);
            it_pcl->z = P_compensate(2);
            if (it_pcl == pcl_out.points.begin()) break;
        }
    }
  }
}


void ImuProcess::Process(const MeasureGroup &meas, StatesGroup &stat, PointCloudXYZI::Ptr cur_pcl_un_)
{
  if (imu_en)
  {
    if(meas.imu.empty())  return;
    ROS_ASSERT(meas.lidar != nullptr);

    // cout << "imu_need_init_: " << imu_need_init_ << endl;

    if (imu_need_init_)   //初始为true
    {
      cout << "LI_init_done: " << LI_init_done << endl;
        if(!LI_init_done){    //如果未完成 LI init
            /// The very first lidar frame
            IMU_init(meas, stat, init_iter_num);  //如果开头几帧  需要初始化IMU参数
            cout << "init_iter_num: " << init_iter_num <<endl;
            imu_need_init_ = true;
            last_imu_   = meas.imu.back();
            if (init_iter_num > MAX_INI_COUNT)
            {
                cov_acc *= pow(G_m_s2 / mean_acc.norm(), 2);
                imu_need_init_ = false;

                cov_acc = cov_acc_scale;
                cov_gyr = cov_gyr_scale;

                ROS_INFO("IMU Initialization Done: Gravity: %.4f %.4f %.4f, Acc norm: %.4f", stat.gravity[0], stat.gravity[1], stat.gravity[2], mean_acc.norm());
                IMU_mean_acc_norm = mean_acc.norm();
            }
        }
        else{    //如果完成 LI init， 转到LIO  online refinement   imu_need_init_ = false
            cout << endl;
            printf(BOLDMAGENTA "1 [Refinement] Switch to LIO mode, online refinement begins.\n\n" RESET);
            last_imu_   = meas.imu.back();
            imu_need_init_ = false;
            cov_acc = cov_acc_scale;
            cov_gyr = cov_gyr_scale;
        }
        return;
    }
    propagation_and_undist(meas, stat, *cur_pcl_un_);   //有IMU的前向传播   类似fast-lio
  }
  else
  {
     Forward_propagation_without_imu(meas, stat, *cur_pcl_un_);   //没有IMU的前向传播   在没有init完成时  执行的是这个
  }
}
#endif