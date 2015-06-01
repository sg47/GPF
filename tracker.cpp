#include "utils.h"
#include <sys/stat.h>


class Tracker {
public:
    enum STATE_DOMAIN {
        Aff2, SL3, SE3
    };
    struct Inputs {
        cv::Mat I; // image
        cv::Mat D; // depth
        cv::Mat C; // point cloud
    };
    struct SSDParams {
        cv::Mat R;
    };
    struct PCAParams {
        cv::Mat R;
        int n_basis;
        int update_interval;
        float forget_factor;
        
    };
    struct Params {
        std::string id; 
        
        std::string frame_dir;
        std::string file_fmt;
        int init_frame;
        int start_frame;
        int end_frame;
        float fps;
        cv::Mat K;
        
        int n_particles;
        std::vector<float> template_xs;
        std::vector<float> template_ys;
        STATE_DOMAIN state_domain;
        std::vector<cv::Mat> E;
        cv::Mat P;
        cv::Mat Q;
        float a;
        
        std::string obs_mdl_type;
        SSDParams ssd_params;
        PCAParams pca_params;
    };
    struct State {
        cv::Mat X;
    };
    struct AR {
        cv::Mat A;
    };
    struct Particle {
        struct State state;
        struct AR    ar;
    };
private:
    Tracker::Params         params_;
    Tracker::State          state_;
    std::vector<Particle>   particles_;
    std::vector<float>      w_;
    std::vector<cv::Mat>    tracks_;
    cv::Mat                 template_pnts_;
    cv::Mat                 template_poly_pnts_;
    
    std::string             obs_mdl_type_;
    cv::Mat                 pca_basis_;
    cv::Mat                 pca_mean_;
    cv::Mat                 pca_svals_;
    int                     pca_n_;
    
    double                  time_prev_sec_;
    
    // buffers
    std::vector<Tracker::Particle> particles;
    std::vector<Tracker::Particle> particles_res;
    std::vector<float> w;
    std::vector<float> w_res;
private:
    void warp_template(const cv::Mat &i_I, const cv::Mat &i_X, const cv::Mat &i_template_pnts, cv::Mat &o_warped_I) {
        // warp points
        cv::Mat warped_pnts;
        if( params_.state_domain == STATE_DOMAIN::SE3 ) {
            cv::Mat X_rot = i_X * i_template_pnts;
            warped_pnts = params_.K * X_rot.rowRange(0, 3);
        } else {
            warped_pnts = i_X * i_template_pnts;
        }
        // convert to int
        std::vector<int> c_ind, r_ind, valid;
        for(int i=0; i<warped_pnts.cols; ++i) {
            int c = std::round(warped_pnts.at<float>(0, i) / warped_pnts.at<float>(2, i));
            int r = std::round(warped_pnts.at<float>(1, i) / warped_pnts.at<float>(2, i));
            c_ind.push_back(c);
            r_ind.push_back(r);
            if( c < 0 || i_I.cols-1 < c || r < 0 || i_I.rows-1 < r ) {
                valid.push_back(0);
                std::runtime_error("assumes all points are valid.");
            } else
                valid.push_back(1);
        }
        // copy values
        if( i_I.channels() != 1)
            std::runtime_error("assume a gray image");
        cv::Mat warped_I;
        for( int i=0; i<c_ind.size(); ++i) {
            int c = std::min(std::max(0, c_ind[i]), i_I.cols-1);
            int r = std::min(std::max(0, r_ind[i]), i_I.rows-1);
            warped_I.push_back(i_I.at<float>(r, c));
        }
        // return
        o_warped_I = warped_I;
    }
    void learn_obs_mdl(const cv::Mat &i_I, const cv::Mat &i_X) {
        int init_size = 5;
        // obtain/save the corresponding template
        cv::Mat warped_I;
        warp_template(i_I, i_X, template_pnts_, warped_I);
        tracks_.push_back(warped_I);
        std::cerr << "- tracks_.size() = " << tracks_.size() << std::endl;
        // choose the obs mdl
        obs_mdl_type_ = params_.obs_mdl_type;
        if( obs_mdl_type_.compare("onlinePCA")==0 && tracks_.size() < init_size )
            obs_mdl_type_ = "SSD";
        // update pca
        if( obs_mdl_type_.compare("onlinePCA")==0 ) {
            if( pca_basis_.cols == 0 ) {
                // initiate
                // X
                cv::Mat X(tracks_[0].rows, tracks_.size(), CV_32F);
                for( int i=0; i<tracks_.size(); ++i )
                    tracks_[i].col(0).copyTo(X.col(i));
                std::cerr << "- size(X), should be n x 2: " << X.rows << ", " << X.cols << std::endl;
                // Xbar
                cv::Mat Xbar;
                cv::reduce(X, Xbar, 1, CV_REDUCE_AVG);
                // A = X - Xbar
                cv::Mat A(X.rows, X.cols, CV_32F);
                for( int i=0; i<X.cols; ++i )
                    A.col(i) = X.col(i) - Xbar;
                // svd(A)
                cv::Mat U, Vt, sv;
                cv::SVD::compute(A, sv, U, Vt);
                std::cerr << "- size(U): " << U.rows << ", " << U.cols << std::endl;
                std::cerr << "- sv: " << sv << std::endl;
                // keep valid basis
                int n_val_basis = std::min(params_.pca_params.n_basis, U.cols);
                int n_eff_basis = -1;
                float eff_thres = 0.95;
                cv::Mat sv_sum;
                cv::Mat sv_cumsum(1, 1, CV_32F, cv::Scalar(0));
                cv::reduce(sv, sv_sum, 0, CV_REDUCE_SUM);
                for( int i=0; i<sv.rows; ++i ) {
                    sv_cumsum += sv.at<float>(i);
                    float thes = sv_cumsum.at<float>(0) / sv_sum.at<float>(0);
                    if( thes > eff_thres ) {
                        n_eff_basis = i + 1;
                        break;
                    }
                }
                n_val_basis = std::min( n_val_basis, n_eff_basis );
                std::cerr << "- n_val_basis: " << n_val_basis << std::endl;
                // save
                if( n_val_basis > 0 ) {
                    U.colRange(0, n_val_basis).copyTo(pca_basis_);
                    sv.rowRange(0, n_val_basis).copyTo(pca_svals_);
                    pca_mean_ = Xbar;
                    pca_n_ = X.cols;
                    
                    std::cerr << "- initial pca_svals_: " << pca_svals_ << std::endl;
                    std::cerr << "- initial size(pca_basis_): " << pca_basis_.rows << ", " << pca_basis_.cols << std::endl;
                    std::cerr << "- initial pca_basis_[0]: " << pca_basis_.at<float>(0) << std::endl;
                    std::cerr << "- initial pca_n_: " << pca_n_ << std::endl;
                    std::cerr << "- initial size(pca_mean_): " << pca_mean_.rows << ", " << pca_mean_.cols << std::endl;
                    std::cerr << "- initial pca_mean_[0]: " << pca_mean_.at<float>(0) << std::endl;
                }
            } else {
                // update
                if( (tracks_.size()-init_size) % params_.pca_params.update_interval == 0 ) {
                    int m = params_.pca_params.update_interval;
                    float ff = params_.pca_params.forget_factor;
                    const cv::Mat M = pca_mean_;
                    const cv::Mat U = pca_basis_;
                    const cv::Mat sv = pca_svals_;
                    const int n = pca_n_;
                    // B
                    cv::Mat B(tracks_[0].rows, m, CV_32F);
                    for(int i=tracks_.size()-m; i<tracks_.size(); ++i) 
                        tracks_[i].col(0).copyTo(B.col(i-(tracks_.size()-m)));
                    std::cerr << "- size(X), should be n x 2: " << B.rows << ", " << B.cols << std::endl;
                    // M_B
                    cv::Mat M_B;
                    cv::reduce(B, M_B, 1, CV_REDUCE_AVG);
                    // M_C
                    cv::Mat M_C = ff*float(n)/(ff*float(n) + float(m))*M + float(m)/(ff*float(n) + float(m))*M_B;
                    // Bn
                    cv::Mat Bn(B.rows, B.cols, CV_32F);
                    for(int i=0; i<B.cols; ++i)
                        Bn.col(i) = B.col(i) - M_B;
                    // Badd
                    cv::Mat Badd = std::sqrt(float(n)*float(m)/(float(n)+float(m))) * (M_B - M);
                    // Bh
                    cv::Mat Bh;
                    cv::hconcat(Bn, Badd, Bh);
                    // Bt
                    cv::Mat Bt_i = Bh - U * U.t() * Bh;
                    cv::Mat Bt;
                    qr_thin(Bt_i, Bt);
                    std::cerr << "- Bt.size(): " << Bt.rows << ", " << Bt.cols << std::endl;
                    // R
                    cv::Mat R, UB1, UB2, UB, LB1, LB2, LB;
                    UB1 = ff * cv::Mat::diag(sv);
                    UB2 = U.t() * Bh;
                    cv::hconcat(UB1, UB2, UB);
                    LB1 = cv::Mat::zeros(Bt.cols, sv.rows, CV_32F);
                    LB2 = Bt.t() * Bt_i;
                    cv::hconcat(LB1, LB2, LB);
                    R.push_back(UB);
                    R.push_back(LB);
                    std::cerr << "- R.size(): " << R.rows << ", " << R.cols << std::endl;
                    // svd(R)
                    cv::Mat svt, Ut, Vtt;
                    cv::SVD::compute(R, svt, Ut, Vtt);
                    std::cerr << "- S of svd(R): " << svt << std::endl;
                    // keep valid basis
                    int n_val_basis = std::min(params_.pca_params.n_basis, Ut.cols);
                    int n_eff_basis = -1;
                    float eff_thres = 0.95;
                    cv::Mat sv_sum;
                    cv::Mat sv_cumsum(1, 1, CV_32F, cv::Scalar(0));
                    cv::reduce(svt, sv_sum, 0, CV_REDUCE_SUM);
                    for( int i=0; i<svt.rows; ++i ) {
                        sv_cumsum += svt.at<float>(i);
                        float thes = sv_cumsum.at<float>(0) / sv_sum.at<float>(0);
                        if( thes > eff_thres ) {
                            n_eff_basis = i + 1;
                            break;
                        }
                    }
                    n_val_basis = std::min( n_val_basis, n_eff_basis );
                    std::cerr << "- n_val_basis: " << n_val_basis << std::endl;
                    // save
                    cv::Mat UBt, U_C, sv_C;
                    cv::hconcat(U, Bt, UBt);
                    U_C = UBt * Ut.colRange(0, n_val_basis);
                    sv_C = svt.rowRange(0, n_val_basis);
                    
                    U_C.copyTo(pca_basis_);
                    sv_C.copyTo(pca_svals_);
                    pca_mean_ = M_C;
                    pca_n_ = m + n;
                    
                    std::cerr << "- updated pca_svals_: " << pca_svals_ << std::endl;
                    std::cerr << "- updated size(pca_basis_): " << pca_basis_.rows << ", " << pca_basis_.cols << std::endl;
                    std::cerr << "- updated pca_basis_[0]: " << pca_basis_.at<float>(0) << std::endl;
                    std::cerr << "- updated pca_n_: " << pca_n_ << std::endl;
                    std::cerr << "- updated size(pca_mean_): " << pca_mean_.rows << ", " << pca_mean_.cols << std::endl;
                    std::cerr << "- updated pca_mean_[0]: " << pca_mean_.at<float>(0) << std::endl;
                }
            }
            
        }
        
        
    }
    void eval_obs_mdl(const cv::Mat &i_I, const cv::Mat &i_X, cv::Mat &o_dist, cv::Mat &o_R) {
        
        cv::Mat warped_I;
        warp_template(i_I, i_X, template_pnts_, warped_I);
        if( obs_mdl_type_.compare("SSD") == 0) { 
            // SSD
            cv::Mat diff = warped_I - tracks_[0];
            // return
            if( diff.rows < diff.cols )
                std::runtime_error("- invalid shape of diff");
            float len = float(diff.rows);
            float dist = diff.dot(diff) / len;
            o_dist = cv::Mat(1, 1, CV_32F, cv::Scalar(dist));
            o_R = params_.ssd_params.R;
        } else if( obs_mdl_type_.compare("onlinePCA") == 0) {
            const cv::Mat M = pca_mean_;
            const cv::Mat U = pca_basis_;
            const cv::Mat sv = pca_svals_;
            // Xmn
            cv::Mat Xmn = warped_I - M;
            cv::Mat Xmnproj = U.t() * Xmn;
            // y1
            double e1 = Xmn.dot(Xmn);
            double e2 = Xmnproj.dot(Xmnproj);
            cv::Mat y1(1, 1, CV_32F, e1 - e2);
            //y1 = y1 / M.rows;
            // y2
            cv::Mat ev, evinv, Xmnproj2;
            cv::pow(sv, 2, ev);
            cv::divide(1, ev, evinv);
            cv::pow(Xmnproj, 2, Xmnproj2);
            cv::Mat y2 = evinv.t() * Xmnproj2;
            // e
            o_dist.push_back(y1);
            o_dist.push_back(y2);
            // R
            o_R = params_.pca_params.R;
            
        }
        // need more accuracy
        o_dist.convertTo(o_dist, CV_64F);
        o_R.convertTo(o_R, CV_64F);
    }
    
    void propose_particles(std::vector<Tracker::Particle> &i_particles, std::vector<Tracker::Particle> &o_particles) {
        // importance density: P(X_t | X_{t-1}) = N_{aff}( f(X_{t-1}), Q), Q = P * dt
        int n_particles = i_particles.size();
        o_particles.resize(n_particles);
        for( int i=0; i<n_particles; ++i ) {
            // f(X_{t-1})
            cv::Mat X_prev = i_particles[i].state.X;
            cv::Mat A_prev = i_particles[i].ar.A;
            cv::Mat f_X_prev;
            transit_state(X_prev, A_prev, f_X_prev);
            // X_t
            cv::Mat X;
            switch(params_.state_domain) {
                case STATE_DOMAIN::Aff2:
                    mvnrnd_aff2(f_X_prev, params_.Q, params_.E, X);
                    break;
                case STATE_DOMAIN::SL3:
                    mvnrnd_sl3(f_X_prev, params_.Q, params_.E, X);
                    break;
                case STATE_DOMAIN::SE3:
                    mvnrnd_se3(f_X_prev, params_.Q, params_.E, X);
                    break;
            }
            // A_t
            cv::Mat XX = X_prev.inv() * X;
            cv::Mat A;
            logm(XX, A);
            A *= params_.a;
            // save the candidate particle
            o_particles[i].state.X = X;
            o_particles[i].ar.A = A;
        }
    }
    
    void transit_state(cv::Mat &i_X, cv::Mat &i_A, cv::Mat &o_X) {
        // expm
        cv::Mat Aexp;
        expm(i_A, Aexp);
        // X * expm(A)
        o_X = i_X * Aexp;
        o_X /= o_X.at<float>(o_X.rows-1, o_X.cols-1); //FIXME: okay?
        if( params_.state_domain == STATE_DOMAIN::SL3 ) {
            // det(X) = 1
            o_X = o_X / std::pow( cv::determinant(o_X), 1/o_X.rows );
        }
            
    }
    
    void update_particle_weights( std::vector<Tracker::Particle> &i_particles, const cv::Mat &i_I, std::vector<float> &i_w, std::vector<float> &o_w) {
        // P(Y_t | X_t)
        int n_particles = i_particles.size();
        o_w.resize(n_particles);
        float w_sum = 0, prob;
        cv::Mat prob_n, prob_dn;
        for( int i=0; i<n_particles; ++i ) {
            cv::Mat dist, R;
            eval_obs_mdl(i_I, i_particles[i].state.X, dist, R);
            cv::exp( - dist.t() * R.inv() * dist / 2, prob_n );
            cv::sqrt( 2 * 3.14 * cv::determinant(R), prob_dn );
            //prob = prob_n.at<double>(0) / prob_dn.at<double>(0);
            prob = cv::exp( - dist.at<double>(0) / R.at<double>(0, 0) * dist.at<double>(0) / 2);
//             std::cout << "dist(1) = " << dist.at<double>(0) << std::endl;
//             std::cout << "dist(2) = " << dist.at<double>(1) << std::endl;
//             std::cout << "prob(1) = " << cv::exp( - dist.at<double>(0) / R.at<double>(0, 0) * dist.at<double>(0) / 2) << std::endl;
//             std::cout << "prob(2) = " << cv::exp( - dist.at<double>(1) / R.at<double>(1, 1) * dist.at<double>(1) / 2) << std::endl;
//             std::cout << "prob = " << prob << std::endl << std::endl;
            o_w[i] = i_w[i] * prob;
            w_sum += o_w[i];
        }
        if(w_sum == 0) {
            std::cout << "w_sum == 0" << std::endl;
            cv::waitKey(0);
            std::runtime_error("the sum of particle weights are zero!");
        }
        // normalize
        for( int i=0; i<n_particles; ++i ) {
            o_w[i] /= w_sum;
        }
    }
    
    void resample_particles( std::vector<Tracker::Particle> &i_particles, std::vector<float> &i_w, 
            std::vector<Tracker::Particle> &o_particles, std::vector<float> &o_w ) {
        int n_particles = i_particles.size();
        o_particles.resize(n_particles);
        o_w.resize(n_particles);
        // resample
        std::vector<int> index;
        resample(i_w, n_particles, n_particles, index);
        // return
        float w_sum = 0;
        for( int i=0; i<n_particles; ++i ) {
            o_particles[i] = i_particles[index[i]];
            o_w[i] = i_w[index[i]];
            w_sum += o_w[i];
        }
        // normalize
        for( int i=0; i<n_particles; ++i ) {
            o_w[i] /= w_sum;
        }
    }
    
    void find_X_opt( std::vector<Tracker::Particle> &i_particles, std::vector<float> &i_w, cv::Mat &o_X_opt ) {
        switch(params_.state_domain) {
            case STATE_DOMAIN::Aff2:
            {
                int n_particles = i_particles.size();
                // find max ind
                float val_max = i_w[0];
                int ind_max = 0;
                for( int i=1; i<n_particles; ++i ) {
                    if( val_max < i_w[i] ) {
                        val_max = i_w[i];
                        ind_max = i;
                    }
                }
                std::cerr << "- max particle val: " << val_max << std::endl;
                std::cerr << "- max particle ind: " << ind_max << std::endl;
                // G_max
                cv::Mat X_max = i_particles[ind_max].state.X;
                cv::Mat G_max(2, 2, CV_32F, cv::Scalar(0));
                G_max.at<float>(0, 0) = X_max.at<float>(0, 0);
                G_max.at<float>(0, 1) = X_max.at<float>(0, 1);
                G_max.at<float>(1, 0) = X_max.at<float>(1, 0);
                G_max.at<float>(1, 1) = X_max.at<float>(1, 1);
                cv::Mat G_max_inv = G_max.inv();
                // U
                cv::Mat U(2, 2, CV_32F, cv::Scalar(0)), U_tmp;
                cv::Mat G(2, 2, CV_32F, cv::Scalar(0)), G_tmp; //FIXME: inefficient
                for( int i=0; i<n_particles; ++i ) {
                    cv::Mat X = i_particles[i].state.X;
                    G.at<float>(0, 0) = X.at<float>(0, 0);
                    G.at<float>(0, 1) = X.at<float>(0, 1);
                    G.at<float>(1, 0) = X.at<float>(1, 0);
                    G.at<float>(1, 1) = X.at<float>(1, 1);
                    G_tmp = G_max_inv * G;
                    logm( G_tmp, U_tmp );
                    U += U_tmp / float(n_particles); 
                }
                // G_bar
                expm(U, U_tmp);
                cv::Mat G_bar = G_max * U_tmp;
                // t_bar
                float t1 = 0, t2 = 0;
                for( int i=0; i<n_particles; ++i ) {
                    t1 += i_particles[i].state.X.at<float>(0, 2) / float(n_particles);
                    t2 += i_particles[i].state.X.at<float>(1, 2) / float(n_particles);
                }
                // return
                o_X_opt = cv::Mat(3, 3, CV_32F, cv::Scalar(0));
                o_X_opt.at<float>(0, 0) = G_bar.at<float>(0, 0);
                o_X_opt.at<float>(0, 1) = G_bar.at<float>(0, 1);
                o_X_opt.at<float>(1, 0) = G_bar.at<float>(1, 0);
                o_X_opt.at<float>(1, 1) = G_bar.at<float>(1, 1);
                o_X_opt.at<float>(0, 2) = t1;
                o_X_opt.at<float>(1, 2) = t2;
                o_X_opt.at<float>(2, 2) = 1;
            }
            break;
                
            case STATE_DOMAIN::SL3:
            {
                double thres = 1e-4;
                int n_particles = i_particles.size();
                cv::Mat mu;
                i_particles[0].state.X.copyTo(mu);
                do {
                    cv::Mat mu_inv = mu.inv();
                    cv::Mat log_sum(mu.rows, mu.cols, CV_32F, cv::Scalar(0));
                    for( int i=0; i<n_particles; ++i ) {
                        cv::Mat dX = mu_inv * i_particles[i].state.X;
                        cv::Mat log_dX;
                        logm(dX, log_dX);
                        log_sum += log_dX;
                    }
                    cv::Mat dmu;
                    log_sum /= float(n_particles);
                    expm(log_sum, dmu);
                    mu *= dmu;
                    cv::Mat log_dmu;
                    logm(dmu, log_dmu);
                    if( cv::norm( log_dmu.reshape(log_dmu.rows*log_dmu.cols) ) < thres )
                        break;
                } while(true);
                
                // return
                o_X_opt = mu;
                o_X_opt /= o_X_opt.at<float>(o_X_opt.rows-1, o_X_opt.cols-1); //FIXME: okay?
                // det(X) = 1
                o_X_opt = o_X_opt / std::pow( cv::determinant(o_X_opt), 1/o_X_opt.rows );
            }
            break;

            case STATE_DOMAIN::SE3:
            {        
                int n_particles = i_particles.size();
                // arithmetic mean of R, R_am
                cv::Mat R_am(3, 3, CV_32F, cv::Scalar(0));
                for( int r=0; r<R_am.rows; ++r )
                    for( int c=0; c<R_am.cols; ++c )
                        for( int i=0; i<n_particles; ++i )
                            R_am.at<float>(r, c) += i_particles[i].state.X.at<float>(r, c) / float(n_particles);
                // R_m
                cv::Mat R_amt = R_am.t();
                float d_a[9] = {1, 0, 0, 0, 1, 0, 0, 0, -1};
                cv::Mat d(3, 3, CV_32F, d_a);
                cv::Mat U, Vt, sv, R_m;
                cv::SVD::compute(R_amt, sv, U, Vt);
                if( cv::determinant(R_amt) > 0 )
                    R_m = Vt.t() * U.t();
                else
                    //Rm = V * diag([1 1 -1]) * U';
                    R_m = Vt.t() * d * U.t();

                // arithmetic mean of t, t_am
                cv::Mat t(3, 1, CV_32F, cv::Scalar(0));
                for( int i=0; i<n_particles; ++i ) {
                    t.at<float>(0, 0) += i_particles[i].state.X.at<float>(0, 3) / float(n_particles);
                    t.at<float>(1, 0) += i_particles[i].state.X.at<float>(1, 3) / float(n_particles);
                    t.at<float>(2, 0) += i_particles[i].state.X.at<float>(2, 3) / float(n_particles);
                }
                // return
                cv::Mat last_row(1, 4, CV_32F, cv::Scalar(0));
                last_row.at<float>(0, 3) = 1;
                cv::Mat X_opt;
                cv::hconcat(R_m, t, X_opt);
                X_opt.push_back(last_row);
                o_X_opt = X_opt;
            }
            break;
        }
    }
     
    void load_params(const std::string &i_conf_fn) {
        std::ifstream conf_file(i_conf_fn);
        std::string line;  
        // proj name
        while(std::getline(conf_file, line)) {
            if(line[0] == '#' || line.size() == 0) continue;
            else {
                params_.id = line;
                break;
            }
        }
        
        // frame_dir
        while(std::getline(conf_file, line)) {
            if(line[0] == '#' || line.size() == 0) continue;
            else {
                params_.frame_dir = line;
                break;
            }
        }
        // file_fmt
        while(std::getline(conf_file, line)) {
            if(line[0] == '#' || line.size() == 0) continue;
            else {
                params_.file_fmt = line;
                break;
            }
        }
        
        // init_frame
        while(std::getline(conf_file, line)) {
            if(line[0] == '#' || line.size() == 0) continue;
            else {
                std::istringstream iss(line);
                iss >> params_.init_frame;
                break;
            }
        }
        // start_frame
        while(std::getline(conf_file, line)) {
            if(line[0] == '#' || line.size() == 0) continue;
            else {
                std::istringstream iss(line);
                iss >> params_.start_frame;
                break;
            }
        }
        // end_frame
        while(std::getline(conf_file, line)) {
            if(line[0] == '#' || line.size() == 0) continue;
            else {
                std::istringstream iss(line);
                iss >> params_.end_frame;
                break;
            }
        }
        // fps
        while(std::getline(conf_file, line)) {
            if(line[0] == '#' || line.size() == 0) continue;
            else {
                std::istringstream iss(line);
                iss >> params_.fps;
                break;
            }
        }    
        // K
        while(std::getline(conf_file, line)) {
            if(line[0] == '#' || line.size() == 0) continue;
            else {
                std::istringstream iss(line);
                std::vector<float> K_vec(9);
                    for(int i=0; i<9; ++i)
                        iss >> K_vec[i];
                    cv::Mat K(3, 3, CV_32F);
                    int i_vec = 0;
                    for(int i=0; i<3; ++i)
                        for(int j=0; j<3; ++j)
                            K.at<float>(i, j) = K_vec[i_vec++];
                    params_.K = K;
                break;
            }
        }
        // n_particles
        while(std::getline(conf_file, line)) {
            if(line[0] == '#' || line.size() == 0) continue;
            else {
                std::istringstream iss(line);
                iss >> params_.n_particles;
                break;
            }
        }
        // state_domain
        while(std::getline(conf_file, line)) {
            if(line[0] == '#' || line.size() == 0) continue;
            else {
                if( line.compare("Aff2") == 0)
                    params_.state_domain = STATE_DOMAIN::Aff2;
                else if( line.compare("SL3") == 0)
                    params_.state_domain = STATE_DOMAIN::SL3;
                else if( line.compare("SE3") == 0)
                    params_.state_domain = STATE_DOMAIN::SE3;
                break;
            }
        }
        // read basis
        int n_basis, n_dim;
        switch(params_.state_domain) {
            case STATE_DOMAIN::Aff2:
                n_basis = 6, n_dim = 3;
                break;
            case STATE_DOMAIN::SL3:
                n_basis = 8, n_dim = 3;
                break;
            case STATE_DOMAIN::SE3:
                n_basis = 6, n_dim = 4;
                break;
        }
        for(int E_ind=0; E_ind<n_basis; ++E_ind)
        {
            while(std::getline(conf_file, line)) {
                if(line[0] == '#' || line.size() == 0) continue;
                else {
                    std::istringstream iss(line);
                    std::vector<float> E_vec( n_dim * n_dim );
                    for(int i=0; i<n_dim*n_dim; ++i)
                        iss >> E_vec[i];
                    cv::Mat E(n_dim, n_dim, CV_32F);
                    int i_vec = 0;
                    for(int i=0; i<n_dim; ++i)
                        for(int j=0; j<n_dim; ++j)
                            E.at<float>(i, j) = E_vec[i_vec++];
                    params_.E.push_back(E);
                    break;
                }
            }
        }
        // a
        while(std::getline(conf_file, line)) {
            if(line[0] == '#' || line.size() == 0) continue;
            else {
                std::istringstream iss(line);
                iss >> params_.a;
                break;
            }
        }
        // state_sigs
        while(std::getline(conf_file, line)) {
            if(line[0] == '#' || line.size() == 0) continue;
            else {
                std::istringstream iss(line);
                std::vector<float> sigs(n_basis);
                for( int i=0; i<n_basis; ++i )
                    iss >> sigs[i];
                cv::Mat P_sigs(n_basis, 1, CV_32F);
                for( int i=0; i<n_basis; ++i )
                    P_sigs.at<float>(i) = sigs[i];
                cv::Mat P_sigssq;
                cv::pow(P_sigs, 2, P_sigssq);
                params_.P = cv::Mat::diag(P_sigssq); 
                params_.Q = params_.P / params_.fps; 
                break;
            }
        }
        // obs_mdl_type
        while(std::getline(conf_file, line)) {
            if(line[0] == '#' || line.size() == 0) continue;
            else {
                params_.obs_mdl_type = line;
                break;
            }
        }
        // R_SSD
        while(std::getline(conf_file, line)) {
            if(line[0] == '#' || line.size() == 0) continue;
            else {
                std::istringstream iss(line);
                float R_SSD;
                iss >> R_SSD;
                params_.ssd_params.R = cv::Mat(1, 1, CV_32F, R_SSD);
                break;
            }
        }
        // R_PCA
        while(std::getline(conf_file, line)) {
            if(line[0] == '#' || line.size() == 0) continue;
            else {
                std::istringstream iss(line);
                std::vector<float> R_PCA_vec(4);
                for( int i=0; i<4; ++i )
                    iss >> R_PCA_vec[i];
                cv::Mat R_PCA(2, 2, CV_32F);
                int v_ind = 0;
                for( int i=0; i<2; ++i )
                    for( int j=0; j<2; ++j )
                        R_PCA.at<float>(i, j) = R_PCA_vec[v_ind++];
                params_.pca_params.R = R_PCA;
                break;
            }
        }
        // n_basis
        while(std::getline(conf_file, line)) {
            if(line[0] == '#' || line.size() == 0) continue;
            else {
                std::istringstream iss(line);
                iss >> params_.pca_params.n_basis;
                break;
            }
        }
        // update_interval
        while(std::getline(conf_file, line)) {
            if(line[0] == '#' || line.size() == 0) continue;
            else {
                std::istringstream iss(line);
                iss >> params_.pca_params.update_interval;
                break;
            }
        }
        // forget_factor
        while(std::getline(conf_file, line)) {
            if(line[0] == '#' || line.size() == 0) continue;
            else {
                std::istringstream iss(line);
                iss >> params_.pca_params.forget_factor;
                break;
            }
        }
        // template_xys   
        while(std::getline(conf_file, line)) {
            if(line[0] == '#' || line.size() == 0) continue;
            else {
                // xs
                {
                    std::istringstream iss(line);
                    std::vector<float> template_xs;
                    float tmp;
                    while( iss >> tmp )
                        template_xs.push_back(tmp);
                    params_.template_xs = template_xs;
                }
                // ys
                {
                    std::getline(conf_file, line);
                    std::istringstream iss(line);
                    std::vector<float> template_ys;
                    float tmp;
                    while( iss >> tmp )
                        template_ys.push_back(tmp);
                    params_.template_ys = template_ys;
                }
                break;
            }
        }
    }
    
public:
    Tracker(const std::string &i_conf_fn) {
        load_params(i_conf_fn);
    }
    Tracker(const Tracker::Params &i_params) {
        params_ = i_params;
    }
    
    void read_inputs(int i_t, Tracker::Inputs &i_inputs) {
        char input_fn[1024];
        // rgb
        snprintf(input_fn, 1024, params_.file_fmt.c_str(), i_t);
        std::string rgb_fn = params_.frame_dir + std::string(input_fn) + std::string(".png");
        cv::Mat I_ori = cv::imread(rgb_fn.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
        I_ori.convertTo(i_inputs.I, CV_32F, 1.0/255.0);
        
        if( params_.state_domain == STATE_DOMAIN::SE3 ) {
            // depth
            std::string d_fn = params_.frame_dir + std::string(input_fn) + std::string("_depth.png");
            cv::Mat I_ori = cv::imread(rgb_fn.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
            I_ori.convertTo(i_inputs.I, CV_32F, 1.0/255.0);
            cv::Mat d_ori = cv::imread(d_fn.c_str(), CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR );
            d_ori.convertTo(i_inputs.D, CV_32F);
            // point clouds
            depth2pc(i_inputs.D, params_.K, i_inputs.C);
        }
    }
    
    
    void Init(const Tracker::Inputs &i_inputs, cv::Mat &o_X_opt) {
        cv::Mat i_I = i_inputs.I;
        // init buffers
        particles.resize(params_.n_particles);
        particles_res.resize(params_.n_particles);
        w.resize(params_.n_particles);
        w_res.resize(params_.n_particles);
        // create results dir
        mkdir(params_.id.c_str(), S_IRWXU);
        // check time
        clock_t time_prev_ = clock();
        time_prev_sec_ = double(time_prev_) / CLOCKS_PER_SEC;
        // get inital template
        int n_pnts = params_.template_xs.size();
        std::cerr << "- n_pnts = " << n_pnts << std::endl;
        // ginput
        cv::Mat means;
        {
            cv::Mat xs( 1, n_pnts, CV_32F, params_.template_xs.data() );
            cv::Mat ys( 1, n_pnts, CV_32F, params_.template_ys.data() );
            
            cv::Scalar x_mean = cv::mean( xs );
            cv::Scalar y_mean = cv::mean( ys );
            
            if( params_.state_domain == STATE_DOMAIN::SE3 ) {
                int x_c = std::round(x_mean[0]);
                int y_c = std::round(y_mean[0]);
                means = cv::Mat(3, 1, CV_32F);
                means.at<float>(0) = i_inputs.C.at<float>(y_c, x_c, 0);
                means.at<float>(1) = i_inputs.C.at<float>(y_c, x_c, 1);
                means.at<float>(2) = i_inputs.C.at<float>(y_c, x_c, 2);
                
                cv::Mat template_poly_pnts(4, n_pnts, CV_32F);
                for( int i=0; i<n_pnts; ++i ) {
                    int x = std::round(xs.at<float>(i));
                    int y = std::round(ys.at<float>(i));
                    
                    template_poly_pnts.at<float>(0, i) = i_inputs.C.at<float>(y, x, 0) - means.at<float>(0);
                    template_poly_pnts.at<float>(1, i) = i_inputs.C.at<float>(y, x, 1) - means.at<float>(1);
                    template_poly_pnts.at<float>(2, i) = i_inputs.C.at<float>(y, x, 2) - means.at<float>(2);
                    template_poly_pnts.at<float>(3, i) = 1;
                }
                template_poly_pnts_ = template_poly_pnts;
                
            } else {
                means.push_back(float(x_mean[0]));
                means.push_back(float(y_mean[0]));
                
                cv::Mat xs_c = xs.clone();
                cv::Mat ys_c = ys.clone();
                xs_c -= x_mean[0];
                ys_c -= y_mean[0];
                cv::Mat os = cv::Mat::ones(1, xs_c.cols, CV_32F); 
                template_poly_pnts_.push_back(xs_c);
                template_poly_pnts_.push_back(ys_c);
                template_poly_pnts_.push_back(os);
            }
        }
        std::cerr << "- template means = " << means << std::endl;
        std::cerr << "- template_poly_pnts_.rows = " << template_poly_pnts_.rows;
        std::cerr << ", template_poly_pnts_.cols = " << template_poly_pnts_.cols << std::endl;
        std::cerr << "- template_poly_pnts_ = " << template_poly_pnts_ << std::endl;
        // template points
        {
            std::vector<cv::Point> xys(n_pnts);
            for(int i=0; i<n_pnts; ++i) {
                cv::Point p((int)params_.template_xs[i], (int)params_.template_ys[i]);
                xys[i] = p;
            }
            cv::Mat mask( i_I.rows, i_I.cols, CV_8U, cv::Scalar(0) );
            const cv::Point* pnts[1] = {xys.data()};
            const int cnts[1] = {n_pnts};
            cv::fillPoly( mask, pnts, cnts, 1, cv::Scalar(255) );
            
            if( params_.state_domain == STATE_DOMAIN::SE3 ) {
                // 3d points
                cv::Mat template_pnts;
                for( int c=0; c<mask.cols; ++c ) {
                    for( int r=0; r<mask.rows; ++r ) {
                        if( mask.at<unsigned char>(r, c) == 255 ) {
                            float x = i_inputs.C.at<float>(r, c, 0);
                            float y = i_inputs.C.at<float>(r, c, 1);
                            float z = i_inputs.C.at<float>(r, c, 2);
                            cv::Mat p(1, 4, CV_32F);
                            p.at<float>(0, 0) = x - means.at<float>(0);
                            p.at<float>(0, 1) = y - means.at<float>(1);
                            p.at<float>(0, 2) = z - means.at<float>(2);
                            p.at<float>(0, 3) = 1;
                            template_pnts.push_back(p);
                        }
                    }
                }
                template_pnts_ = template_pnts.t();
            } else {
                // 2d points
                cv::Mat r_ind, c_ind;
                cv::Scalar x_mean(means.at<float>(0));
                cv::Scalar y_mean(means.at<float>(1));
                for( int c=0; c<mask.cols; ++c )
                    for( int r=0; r<mask.rows; ++r )
                        if( mask.at<unsigned char>(r, c) == 255 ) {
                            r_ind.push_back((float)r - (float)y_mean[0]);
                            c_ind.push_back((float)c - (float)x_mean[0]);
                        }
                r_ind = r_ind.t();
                c_ind = c_ind.t();
                
                cv::Mat os = cv::Mat::ones(1, c_ind.cols, CV_32F); 
                template_pnts_.push_back(c_ind);
                template_pnts_.push_back(r_ind);
                template_pnts_.push_back(os);
            }
            // WHICH ONE IS CORRECT? MATLB? C?
            std::cerr << "- template_pnts.rows = " << template_pnts_.rows;
            std::cerr << ", template_pnts_cols = " << template_pnts_.cols << std::endl;
            std::cerr << "- template_pnts_(0, 0) = " << template_pnts_.at<float>(0, 0) << std::endl;
            std::cerr << "- template_pnts_(0, 5) = " << template_pnts_.at<float>(0, 5) << std::endl;
        }
        // init X
        int mat_dim = params_.E[0].rows;
        cv::Mat X0(mat_dim, mat_dim, CV_32F, cv::Scalar(0));
        {
            for( int i=0; i<mat_dim; ++i ) {
                X0.at<float>(i, i) = 1;
                X0.at<float>(i, mat_dim-1) = means.at<float>(i);
            }
            X0.at<float>(mat_dim-1, mat_dim-1) = 1;
            state_.X = X0;
        }
        std::cerr << "- X0 = " << X0 << std::endl;
        // init particles
        {
            particles_.resize(params_.n_particles);
            cv::Mat A0 = cv::Mat::eye(mat_dim, mat_dim, CV_32F);
            for(int i=0; i<particles_.size(); ++i) {
                particles_[i].state.X = X0;
                particles_[i].ar.A = A0;
            }
        }
        // init w
        w_ = std::vector<float>(params_.n_particles, 1/float(params_.n_particles));
        // init obs model
        learn_obs_mdl(i_I, state_.X);
        // return
        o_X_opt = state_.X;
    }
    
    void Track(const Tracker::Inputs &i_inputs, const int i_t, cv::Mat &o_X_opt) {
        cv::Mat i_I = i_inputs.I;
        // propose new Xt
        propose_particles(particles_, particles);
        // update weights
        update_particle_weights(particles, i_I, w_, w);
        std::cerr << "- sample w after update_particle_weights(): " << w[0] << std::endl;
        // resampling
        resample_particles(particles, w, particles_res, w_res);
        std::cerr << "- sample w after resample_particles(): " << w_res[0] << std::endl;
        // find the optimum state
        cv::Mat X_opt;
        find_X_opt(particles_res, w_res, X_opt);
        // learn obs mdl
        learn_obs_mdl(i_I, X_opt);
        // update X, particles, w
        state_.X = X_opt;
        particles_ = particles_res;
        w_ = w_res;
        // return
        o_X_opt = X_opt;
        std::cerr << "- X_opt: " << X_opt << std::endl;
    }
    
    void Show(const Tracker::Inputs &i_inputs, const cv::Mat &i_X, const int i_t, const std::string i_fn_out) {
        cv::Mat i_I = i_inputs.I;
        // check time
        clock_t time_cur = clock();
        double time_cur_sec = double(time_cur) / CLOCKS_PER_SEC;
        double fps = 1.0 / (time_cur_sec - time_prev_sec_);
        time_prev_sec_ = time_cur_sec;
        // I
        cv::Mat I;
        i_I.convertTo(I, CV_8U, 255);
        // draw X
        cv::Mat poly;
        if( params_.state_domain == STATE_DOMAIN::SE3 ) {
            cv::Mat X_rot = i_X * template_poly_pnts_;
            poly = params_.K * X_rot.rowRange(0, 3);
        } else { 
            poly = i_X * template_poly_pnts_;
        }
        std::vector<cv::Point> pnts;
        for( int i=0; i<poly.cols; ++i ) {
            cv::Point p( 
                    (int)std::round(poly.at<float>(0, i)/poly.at<float>(2, i)), 
                    (int)std::round(poly.at<float>(1, i)/poly.at<float>(2, i)));
            pnts.push_back(p);
        }
        const cv::Point *pts[1] = {pnts.data()};
        const int npts[1] = {(int)pnts.size()};
        polylines(I, pts, npts, 1, true, cv::Scalar(255, 0, 0), 1);
        // draw t ::FIXME: access to private vars
        int n_particles = particles_.size();
        float val_max = w_[0];
        int ind_max = 0;
        for( int i=1; i<n_particles; ++i ) {
            if( val_max < w_[i] ) {
                val_max = w_[i];
                ind_max = i;
            }
        }
        for( int i=0; i<n_particles; ++i ) {
            float tx, ty;
            if( params_.state_domain == STATE_DOMAIN::SE3 ) {
                cv::Mat t = particles_[i].state.X.col(3);
                cv::Mat t_proj = params_.K * t.rowRange(0, 3);
                tx = t_proj.at<float>(0) / t_proj.at<float>(2);
                ty = t_proj.at<float>(1) / t_proj.at<float>(2);
            } else {
                tx = particles_[i].state.X.at<float>(0, 2) / particles_[i].state.X.at<float>(2, 2);
                ty = particles_[i].state.X.at<float>(1, 2) / particles_[i].state.X.at<float>(2, 2);
            }
            cv::Point p(tx, ty);
            int r = std::max(1.0, w_[i] / val_max * 3.0);
            circle(I, p, r, cv::Scalar(255));
        }
        
        // add a title
        char title[128];
        sprintf(title, "%.02f fps", fps);
        cv::putText(I, title, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255));
        // show
        cv::imshow("tracking...", I);
        cv::waitKey(1);
        // write results
        cv::imwrite(i_fn_out, I);
    }
    
    void Run() {
        // track
        for(int t=params_.start_frame; t<=params_.end_frame; ++t) {
            std::string fn_out = params_.id + "/" + std::to_string(t) + std::string(".png");
            Tracker::Inputs I;
            read_inputs(t, I);
            
            cv::Mat X_opt;
            if( t == params_.init_frame)
                Init(I, X_opt);
            else
                Track(I, t, X_opt);
            
            Show(I, X_opt, t, fn_out);
        }
    }
};


int main(int argc, char* argv[]) {
    Tracker tracker(argv[1]);
    tracker.Run();
    return 0;
}

  